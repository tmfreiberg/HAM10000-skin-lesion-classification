import os
from pathlib import Path
import pandas as pd
import numpy as np
from processing import process
from utils import display, print_header
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import List, Callable, Dict, Union, Tuple
from torchvision.transforms import Compose, Resize, ToTensor
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import json

from collections import OrderedDict

class image_n_label:
    def __init__(
        self,
        df: pd.DataFrame,
        label_codes: dict,
        data_dir: Path,
        transform: Union[None, 
                         transforms.Compose, 
                         List[Callable]] = None,
        Print: Union[None, bool] = None,        
    ) -> (Image, str, str):
        if "label" in df.columns:
            self.df = df[["image_id", "label"]].copy()
        else:
            self.df = df[["image_id"]].copy()
        self.label_codes = label_codes
        self.data_dir = data_dir
        self.transform = transform
        self.Print = Print
        if self.Print is None:
            self.Print = False
        # We need to reset the index because...KeyError....
        self.df.reset_index(inplace=True)
        # But this eventually throws ValueError: cannot insert level_0, already exists...
        # Replacing self.df = df above with self.df = df.copy() seems to resolve this, but this costs time and memory.
        # Probably should think about a more efficient way to do this.
        # Well, we just need image_id and label here, so... just copy those two columns...

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        img_name = self.data_dir.joinpath(self.df.loc[idx, "image_id"] + ".jpg")
        image = Image.open(img_name)
        image_id = self.df.loc[idx, "image_id"]
        
        if self.transform:
                image = self.transform(image)
                
        if "label" in self.df.columns:
            code = self.df.loc[idx, "label"]
            # One-hot encoding the labels
            label = torch.zeros(len(self.label_codes))
            label[code] = 1
            
            if self.Print:
                print(f"image_id, label, ohe-label: {image_id}, {code}, {label}")
        else:
            label = torch.empty(0)
        
        return image, label, image_id
    
def df_from_ids(filenames: Union[None, str, list] = None,
                     multiplicity: Union[None, int] = None,
                     lesion_ids: Union[None, str, list] = None,
                     df: Union[None, pd.DataFrame] = None,
                     one_img_per_lesion: Union[None, bool] = None,) -> pd.DataFrame:
    if filenames is not None:
        filenames = pd.Series(filenames)
        filenames = filenames.apply(lambda x: x[:x.rfind('.')] if '.' in x else x)
        filenames.name = "image_id"
        if multiplicity is not None:
            filenames = filenames.repeat(multiplicity).reset_index(drop=True)
        filenames_df = pd.DataFrame(filenames)
        return filenames_df
    elif lesion_ids is not None:
        assert all(condition for condition in [isinstance(df, pd.DataFrame), 'image_id' in df.columns, 'lesion_id' in df.columns]), "Invalid DataFrame or missing columns"
        if multiplicity is not None and one_img_per_lesion is None:
            one_img_per_lesion = False
        lesion_ids = pd.Series(lesion_ids)
        working_df = df[df['lesion_id'].isin(lesion_ids)].copy()
        if working_df.empty:
            return working_df
        working_df.drop_duplicates(subset='image_id')
        if one_img_per_lesion is None:            
            return working_df
        elif one_img_per_lesion:
            if multiplicity is None:
                return working_df.drop_duplicates(subset='lesion_id')
            else:
                working_df = working_df.drop_duplicates(subset='lesion_id')
                working_df_repeat = working_df.reindex(working_df.index.repeat(multiplicity)).reset_index(drop=True)
            return working_df_repeat
        
        # else... not one_img_per_lesion
        elif multiplicity is None:
            return working_df 
        # else...  not one_img_per_lesion and multiplicity is something               
        else:
            if "num_images" in working_df.columns:
                # We'll drop the column and put it back in in case the numbers don't mean what we think they mean
                working_df.drop('num_images', axis=1, inplace=True)
            working_df.insert(1, "num_images", working_df["lesion_id"].map(working_df["lesion_id"].value_counts()),)

            m = multiplicity
            sample_image_list = []

            working_df['q'], working_df['r'] = divmod(m,df['num_images']) 

            # m = q*num_images + r. We want q copies of each image corresponding to this lesion_id, plus a further one copy of r of them.
            x = working_df.apply(lambda row: [row['image_id']] * row['q'], axis=1)

            # Add these to the list
            sample_image_list.extend([item for sublist in x for item in sublist])

            # Now for the r 'leftover' images for each lesion
            y_df = pd.DataFrame(columns=working_df.columns)
            y_df = working_df.groupby('lesion_id').apply(lambda group: group.sample(n=group['r'].iloc[0])).reset_index(drop=True)
            y = y_df['image_id'].tolist()

            # Add them to the list
            sample_image_list.extend(y)

            sample_image_list_counts = pd.Series(sample_image_list).groupby(pd.Series(sample_image_list)).size().reset_index(name='img_mult')

            # Merge df with sample_image_list_counts based on 'image_id'
            working_df = pd.merge(working_df, sample_image_list_counts, left_on='image_id', right_on='index', how='inner')

            # Expand rows based on 'img_mult' column
            working_df = working_df.loc[working_df.index.repeat(working_df['img_mult'])].reset_index(drop=True)

            # Drop the temporary 'index' columns
            working_df.drop(['index', 'q', 'r', 'img_mult'], axis=1, inplace=True)

            return working_df                    

def get_probabilities(
        df: pd.DataFrame,
        data_dir: Path,
        model_dir: Path,
        model: Union[None, models.ResNet, models.EfficientNet],
        filename: str,
        label_codes: Union[None,dict],
        transform: Union[None, 
                         transforms.Compose, 
                         List[Callable]] = None,
        batch_size: Union[None, int] = None,
        Print: Union[None, bool] = None,
        save_as: Union[None, str] = None,
       ) -> pd.DataFrame:
    
    try:                
        if label_codes is None:
            try:
                vc = df[['label','dx']].value_counts()
                label_codes = {}
                for key, value in vc.index:
                    if key not in label_codes:
                        label_codes[key] = value
                    else:
                        label_codes[key] = 'other'
            except Exception as e:
                print(f"Error reconstructing label codes: enter a label_codes dictionary: {e}")

        if batch_size is None:
            batch_size = 32

        if Print is None:
            Print = False            

        # Define DataLoader for batch processing        
        data = image_n_label(df, label_codes, data_dir, transform, Print)        
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)

        if model is None:
            raise ValueError("Model is not provided.")

        if '.' in filename:
            filename = filename[:filename.rindex('.')] 
        file_path_pth = model_dir.joinpath(filename + ".pth")
        try:
            state_dict = torch.load(file_path_pth) 
            model.load_state_dict(state_dict)            
        except Exception as e:
            print(f"Error loading {file_path_pth}: {e}.")

        # Set the model to evaluation mode
        model.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Use DataParallel for parallel processing (if multiple GPUs are available)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        # Dataframe to store image_id and prediction
        cols = ["image_id"] + ["prob_" + label for label in label_codes.values()]
        image_id_prob_list = []

        softmax = nn.Softmax(dim=1)
        # Iterate through the DataLoader and make predictions
        with torch.no_grad():
            for images, _, image_ids in dataloader: # NB: labels are skipped (_)
                # Send input tensor to device
                images = images.to(device)
                # Make predictions using the model
                outputs = model(images)
                # Apply softmax to get probabilities
                probabilities = softmax(outputs)

                # Move probabilities to CPU before converting to NumPy array
                probabilities_cpu = probabilities.cpu().numpy()

                series_dict = {}
                series_dict["image_id"] = pd.Series(image_ids)

                for idx, label in enumerate(label_codes.values()):
                    series_dict["prob_" + label] = pd.Series(probabilities_cpu[:, idx])

                batch_df = pd.DataFrame(series_dict)
                image_id_prob_list.append(batch_df)
       
        # Concatenate all DataFrames in image_id_prob_list
        image_id_prob = pd.concat(image_id_prob_list, axis=0) 
        # Combine...
        image_id_prob=image_id_prob.reset_index(drop=True)
        df_probabilities = df.merge(image_id_prob, left_index=True, right_index=True)
        if (df_probabilities['image_id_x'] == df_probabilities['image_id_y']).all():
            df_probabilities.drop('image_id_y', axis=1, inplace=True)
            df_probabilities.rename(columns={'image_id_x': 'image_id'}, inplace=True)

        # Save to file (if applicable)
        if save_as is not None:
            save_as = save_as + "_probabilities.csv"
            file_path = model_dir.joinpath(save_as)
            print(f"Saving probabilities: {file_path}")
            df_probabilities.to_csv(file_path)

        return df_probabilities 
        
    except Exception as e:
        print(f"Error in get_probabilities: {e}")
        return df    
    
def final_prediction(raw_probabilities_df: pd.DataFrame,
                     label_codes: Dict[int, str],
                     aggregate_method: Union[None, Dict[str, List[str]]] = None,
                     threshold_dict_help: Union[None, OrderedDict[str, float]] = None,
                     threshold_dict_hinder: Union[None, OrderedDict[str, float]] = None,  
                     votes_to_win_dict: Union[None, OrderedDict[str, int]] = None,
                     prefix: Union[None, str] = None,) -> pd.DataFrame:
    
    if prefix is None:
        prefix = 'prob_'
    
    assert 'image_id' in raw_probabilities_df.columns, "DataFrame must contain image_id column."
    has_prob_column = any(col.startswith(prefix) for col in raw_probabilities_df.columns)
    assert has_prob_column, f"DataFrame must contain column names starting with {prefix}."
    
    step1_df = aggregate_probabilities(df=raw_probabilities_df,
                                        method=aggregate_method,
                                        prefix=prefix,)
    
    step2_df = append_prediction(probabilities_df=step1_df, 
                                 threshold_dict_help=threshold_dict_help,
                                 threshold_dict_hinder=threshold_dict_hinder,
                                 label_codes=label_codes,
                                 prefix=prefix,)
    
    step3_df = aggregate_predictions(df=step2_df, 
                                     pred_col='pred',
                                     label_codes=label_codes,
                                     votes_to_win_dict=votes_to_win_dict,)
    
    return step3_df
    
def aggregate_probabilities(df: pd.DataFrame,
                            method: Union[None, Dict[str, List[str]]] = None,
                            prefix: Union[None, str] = None,
                            ) -> pd.DataFrame:
    
    if method is None:
        return df

    if isinstance(method, dict):
        if 'mean' not in method.keys():
            method['mean'] = []
        if 'max' not in method.keys():
            method['max'] = []
        if 'min' not in method.keys():
            method['min'] = []
        if not any(method[key] for key in method.keys()):
            return df
        
    if prefix is None:
        prefix = 'prob_'
        
    output = df.copy()
       
    method['mean'] = [prefix + lesion for lesion in method['mean'] if prefix + lesion in output.columns]
    method['max'] = [prefix + lesion for lesion in method['max'] if prefix + lesion in output.columns]
    method['min'] = [prefix + lesion for lesion in method['min'] if prefix + lesion in output.columns]
    
    if 'lesion_id' in output.columns:
        group = 'lesion_id'
    elif 'image_id' in output.columns:
        group = 'image_id'
    else:
        print("DataFrame must have \'image_id\' column.")
        return   

    numeric_columns = output.columns[output.columns.str.startswith('prob_')].tolist()
    
    mean_probs = output.groupby(group)[numeric_columns].mean()
    max_probs = output.groupby(group)[numeric_columns].max()
    min_probs = output.groupby(group)[numeric_columns].min()

    for col in mean_probs.columns:
        if col in method['mean']:
            output[col] = output[group].map(mean_probs[col])
        elif col in method['max']:
            output[col] = output[group].map(max_probs[col])
        elif col in method['min']:
            output[col] = output[group].map(min_probs[col])
    
    return output        
    
def append_prediction(probabilities_df: pd.DataFrame, 
                      threshold_dict_help: Union[None, OrderedDict[str, float]] = None,
                      threshold_dict_hinder: Union[None, OrderedDict[str, float]] = None,
                      label_codes: Dict[int, str] = None,
                      prefix: Union[None, str] = None,) -> pd.DataFrame:    
    
    if prefix is None:
        prefix = 'prob_'
    # Make a copy of the original dataframe
    output_df = probabilities_df.copy()
    
    # Apply the 'get_argmax' function to the probabilities dataframe and append the result to the original
    output_df['pred'] = probabilities_df.apply(get_argmax, 
                                               prefix=prefix, 
                                               threshold_dict_help=threshold_dict_help,
                                               threshold_dict_hinder=threshold_dict_hinder,
                                               label_codes=label_codes, 
                                               axis=1)
    return output_df

def get_argmax(row: pd.DataFrame, 
               prefix: Union[None, str]=None, 
               threshold_dict_help: Union[None, OrderedDict[str, float]] = None,
               threshold_dict_hinder: Union[None, OrderedDict[str, float]] = None,
               label_codes=Union[None, Dict[int, str]]) -> Union[int, str]:
    
    if prefix is None:
        prefix = 'prob_'
    # Filter columns based on the prefix
    prob_columns = [col for col in row.index if col.startswith(prefix)] #? why .index and not .columns? No, after applying .groupby, the grroupby column becomes the index of the resulting dataframe or something
    probabilities = row[prob_columns].astype(float)
    
    probabilities = threshold(probabilities=probabilities, 
                              threshold_dict_help=threshold_dict_help,
                              threshold_dict_hinder=threshold_dict_hinder,
                              prefix=prefix,)
        
    max_column = probabilities.idxmax()
    dx = max_column.split('_')[1]  # Split the string and return the second part (after the prefix)
    if label_codes:
        inverse_label_codes = { v : k for k, v in label_codes.items() }
        return inverse_label_codes[dx]  # Return the label if label_codes are provided
    else:
        return dx  # Otherwise, return the code itself
    
def threshold(probabilities: pd.Series, 
              threshold_dict_help: Union[None, OrderedDict[str, float]],
              threshold_dict_hinder: Union[None, OrderedDict[str, float]],
              prefix: Union[None, str] = None) -> pd.Series:   
    
    if prefix is None:
        prefix = 'prob_'
    
    if isinstance(threshold_dict_help, OrderedDict):
        for dx, thres in threshold_dict_help.items():
            if prefix + dx in probabilities.index and probabilities[prefix + dx] > thres:
                probabilities[prefix + dx] = 1
                break
    if isinstance(threshold_dict_hinder, OrderedDict):
        for dx, thres in threshold_dict_hinder.items():
                if prefix + dx in probabilities.index and probabilities[prefix + dx] < thres:
                    probabilities[prefix + dx] = 0
                    break            
    return probabilities
   
def aggregate_predictions(df: pd.DataFrame, 
                          label_codes: Dict[int, str],
                          votes_to_win_dict: Union[None, OrderedDict[str, int]]=None,
                          pred_col: Union[None, str] = None,) -> pd.DataFrame:
    if pred_col is None:
        pred_col = 'pred'
    
    if 'lesion_id' in df.columns:
        group = 'lesion_id'
    elif 'image_id' in df.columns:
        group = 'image_id'
    else:
        print("DataFrame must have image_id column.")
        return
    
    try:
        df['pred_final_tmp'] = df.groupby(group).apply(one_and_win, label_codes, votes_to_win_dict, pred_col)['pred_final_tmp']    
        mode_df = df.groupby(group)['pred_final_tmp'].agg(mode_with_random)
        output = df.merge(mode_df, left_on=group, right_index=True, suffixes=('', '_')).drop('pred_final_tmp', axis=1)
        output = output.rename(columns={'pred_final_tmp_': 'pred_final'})
    except:
        mode_df = df.groupby(group)[pred_col].agg(mode_with_random)
        output = df.merge(mode_df, left_on=group, right_index=True, suffixes=('', '_final'))
    
    if 'pred_final_tmp' in output.columns:
        output = output.drop('pred_final_tmp', axis=1)
    
    return output

def one_and_win(g: pd.DataFrame,                
                label_codes: Dict[int, str],
                votes_to_win_dict: Union[None, OrderedDict[str, int]]=None,
                pred_col: Union[None, str] = None,) -> pd.DataFrame:
    
    if pred_col is None:
        pred_col = 'pred'
    
    if isinstance(votes_to_win_dict, OrderedDict):
        inverse_label_codes = { v : k for k, v in label_codes.items() }
        for dx, votes in votes_to_win_dict.items():
            if (g[pred_col] == inverse_label_codes[dx]).sum() >= votes:
                g['pred_final_tmp'] = inverse_label_codes[dx]
                break
    return g

def mode_with_random(x):
    modes = x.mode()
    if not modes.empty:
        return modes[0]
    else:
        max_count = x.value_counts().max()
        modes = x.value_counts()[x.value_counts() == max_count].index.tolist()
        return np.random.choice(modes)    