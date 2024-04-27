import os
from pathlib import Path
import pandas as pd
import numpy as np
from processing import process
from utils import display
# from IPython.display import display
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import List, Callable, Dict, Union
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
        transform=None,
        Print: bool = False,
    ) -> (Image, str, str):
        self.df = df[["image_id", "label"]].copy()
        self.label_codes = label_codes
        self.data_dir = data_dir
        self.transform = transform
        self.Print = Print
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
        code = self.df.loc[idx, "label"]
        # One-hot encoding the labels
        label = torch.zeros(len(self.label_codes))
        label[code] = 1
        image_id = self.df.loc[idx, "image_id"]

        if self.transform:
            image = self.transform(image)

        if self.Print:
            print(f"image_id, label, ohe-label: {image_id}, {code}, {label}")
        return image, label, image_id


class resnet18:
    def __init__(
        self,
        source: process,
        model_dir: Path,
        transform: Union[None, transforms.Compose, List[Callable]],       
        batch_size: int = 32,
        epochs: int = 10,
        base_learning_rate: float = 0.001,
        filename_stem: str = "rn18",
        filename_suffix: str = "",
        overwrite: Union[bool, None] = None,
        code_test: Union[bool, None] = None,        
        Print: Union[bool,None] = None,
        model=models.resnet18(weights="ResNet18_Weights.DEFAULT"),
        state_dict: Union[
            None, Dict[str, torch.Tensor]
        ] = None,  
        epoch_losses: Union[dict, None] = None,
    ) -> None:

        self.source = source
        self.df = self.source.df
        self.restrict_to = self.source.restrict_to,
        self.remove_if  = self.source.remove_if,
        self.drop_row_if_missing_value_in = self.source.drop_row_if_missing_value_in,
        self.tvr = self.source.tvr,
        self.seed = self.source.seed,
        self.keep_first = self.source.keep_first,
        self.stratified = self.source.stratified, 
                
        self.code_test = code_test
        if self.code_test is None:
            self.code_test = False
        if self.code_test:
            self.df_combined = self.source._df_sample_batch
        else:
            self.df_combined = self.source.df_combined
        self._df_train = self.df_combined[self.df_combined['set'].isin(["t1","ta"])]
        self._df_val = self.df_combined[self.df_combined['set'].isin(["v1","va"])] 
        self.train_one_img_per_lesion = self.source.train_one_img_per_lesion
        self.val_one_img_per_lesion = self.source.val_one_img_per_lesion
        self.val_expansion_factor = self.source.val_expansion_factor
        self.to_classify = self.source.to_classify
        self.sample_size = self.source.sample_size
        self.label_codes = self.source._label_codes
        self.label_dict = self.source._label_dict
        self.num_labels = self.source._num_labels        

        self.data_dir = self.source.data_dir
        self.model_dir = model_dir
        self.transform = transform
        self.batch_size = batch_size
        self.epochs = epochs
        if self.code_test:
            self.epochs = 1
        self.base_learning_rate = base_learning_rate
        self.filename_stem = filename_stem
        self.filename_suffix = filename_suffix
        self.overwrite = overwrite
        if self.overwrite is None:
            self.overwrite = False
        self.Print = Print
        if self.Print is None:
            self.Print = False
        if self.code_test:
            self.Print = True
        self.model = model
        self.state_dict = state_dict
        self.epoch_losses = epoch_losses

        self.construct_filename()        
        self.save_attributes_to_file()

    def construct_filename(self) -> None:
        # To construct a string for the filename (for saving)
        tvcode = ""
        try:
            if "ta" in self._df_train["set"].unique():
                tvcode += "ta"
            else:
                tvcode += "t1"
        except:
            pass
        try:
            if "va" in self._df_val["set"].unique():
                tvcode += "va"
            else:
                tvcode += "v1"
        except:
            pass
        balance_code = ""
        try:
            if self.sample_size is not None:
                balance_code += "bal"
            else:
                pass
        except:
            pass
        testcode = ""
        if self.code_test:
            testcode += "test"        

        # Initial filename without suffix
        base_filename = "_".join([self.filename_stem, tvcode, balance_code, testcode, str(self.epochs) + "e",])

        # Find a unique filename by incrementing a counter
        counter = 0
        while not self.overwrite:
            filename = base_filename + f"_{self.filename_suffix}_{counter:02d}"
            filepath = self.model_dir.joinpath(filename + ".pth")

            # Check if the file already exists
            if not os.path.exists(filepath):
                break  # Unique filename found
            else:
                counter += 1  # Increment counter for next attempt

        # New attribute
        if self.overwrite:            
            self._filename = base_filename + f"_{self.filename_suffix}_{counter:02d}" 
            print(f"Existing files will be overwritten. \nBase filename: {self._filename}")
        else:
            self._filename = filename
            print(f"New files will be created. \nBase filename: {self._filename}")

    def train(self) -> None:
        # Define DataLoader for batch processing
        training_data = image_n_label(
            self._df_train, self.label_codes, self.data_dir, self.transform, self.Print
        )
        dataloader = DataLoader(training_data, batch_size=self.batch_size, shuffle=True)

        # Load the ResNet18 model
        model = self.model
        # Unfreeze all layers for fine-tuning
        for param in model.parameters():
            param.requires_grad = True  # All layers unfrozen for fine-tuning right away

        # Replace the last layer for classification with appropriate number of labels
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(self.label_codes))

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.base_learning_rate)

        # Training loop
        num_epochs = self.epochs
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Create a dictionary to record loss
        loss_dict = {
            "train_loss": (-1) * np.ones(self.epochs),
            "val_loss": (-1) * np.ones(self.epochs),
        }

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for images, labels, _ in dataloader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                if self.Print:
                    print(f"outputs.shape: {outputs.shape}")
                    print(f"loss: {loss}")
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            # Calculate validation loss for the epoch
            epoch_loss = running_loss / len(dataloader)
            # Add it to the dictionary
            loss_dict["train_loss"][epoch] = epoch_loss

            # Validation step
            # Define DataLoader for batch processing for validation set
            validation_data = image_n_label(
                self._df_val,
                self.label_codes,
                self.data_dir,
                self.transform,
                self.Print,
            )
            val_dataloader = DataLoader(
                validation_data, batch_size=self.batch_size, shuffle=False
            )  # No need to shuffle for validation

            # Set model to evaluation mode
            if self.Print:
                print("Validating...")
            model.eval()
            val_running_loss = 0.0
            val_epoch_loss = -1
            with torch.no_grad():  # Disable gradient calculation during validation
                for val_images, val_labels, _ in val_dataloader:
                    val_images, val_labels = val_images.to(device), val_labels.to(
                        device
                    )
                    val_outputs = model(val_images)
                    val_loss = criterion(val_outputs, val_labels)
                    if self.Print:
                        print(f"outputs.shape: {outputs.shape}")
                        print(f"val_loss: {val_loss}")
                    val_running_loss += val_loss.item()

                    # Calculate validation loss for the epoch
                    val_epoch_loss = val_running_loss / len(val_dataloader)
                    # Add it to the dictionary
                    loss_dict["val_loss"][epoch] = val_epoch_loss

            print(
                f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, Validation Loss: {val_epoch_loss:.4f}"
            )

        # Now save the model
        file_path = self.model_dir.joinpath(self._filename + ".pth")
        print(f"Saving model.state_dict() as {file_path}.")
        torch.save(model.state_dict(), file_path)

        print("model.state_dict() can now be accessed through state_dict attribute.")
        print("Train/val losses can now be accessed through epoch_losses attribute.")
        self.epoch_losses = loss_dict.copy()        
        self.state_dict = model.state_dict()
        
        # Save the epoch losses to a text file for later
        file_path = self.model_dir.joinpath(self._filename + "_epoch_losses" + ".json")
        for key, value in loss_dict.items():
            if isinstance(value, np.ndarray):
                loss_dict[key] = value.tolist()
        print(f"Epoch losses dictionary save as {file_path}")
        # Save the dictionary to a JSON file
        with open(file_path, 'w') as json_file:
            json.dump(loss_dict, json_file)

    def inference(
        self,
        df_infer: pd.DataFrame = None,
        filename: str = None,
        Print: bool = False,
        save: bool = False,
    ) -> pd.DataFrame:
        if df_infer is None:
            df_infer = self.df

        # Define DataLoader for batch processing
        inference_data = image_n_label(
            df_infer, self.label_codes, self.data_dir, self.transform, self.Print
        )
        dataloader = DataLoader(
            inference_data, batch_size=self.batch_size, shuffle=False
        )

        model = self.model
        # Load the model
        if self.state_dict is None:
            assert (
                filename is not None
            ), "state_dict attribute is None: provide a filename for loading."
            if filename.endswith(".pth"):
                filename = filename[:-4]
            file_path_pth = self.model_dir.joinpath(filename + ".pth")
            file_path_csv = self.model_dir.joinpath(filename + "_infer.csv")
            try:
                model.load_state_dict(torch.load(file_path_pth))
            except Exception as e:
                print(f"Error loading {file_path_pth}: {e}.")
        else:
            file_path_csv = self.model_dir.joinpath(self._filename + "_infer.csv")

        # Set the model to evaluation mode
        model.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Use DataParallel for parallel processing (if multiple GPUs are available)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        # Dataframe to store image_id and prediction
        cols = ["image_id"] + ["prob_" + label for label in self.label_codes.values()]
        image_id_prob = pd.DataFrame({col_name: pd.NA for col_name in cols}, index=[0])

        softmax = nn.Softmax(dim=1)
        # Iterate through the DataLoader and make predictions
        with torch.no_grad():
            for images, labels, image_ids in dataloader:
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

                for idx, label in enumerate(self.label_codes.values()):
                    series_dict["prob_" + label] = pd.Series(probabilities_cpu[:, idx])

                batch_df = pd.DataFrame(series_dict)

                image_id_prob = pd.concat([image_id_prob, batch_df], axis=0)

        # This dataframe contains "image_id" column and a probability column for each class.
        image_id_prob = image_id_prob.dropna(subset=["image_id"])

        # Merge it with the underlying metadata dataframe (or whatever was passed as df_infer).
        try:
            inference_df = pd.merge(df_infer, image_id_prob, on="image_id", how="left")
        except Exception as e:
            print(f"Error merging inference dataframe with input dataframe: {e}")

        # Add this new dataframe with inferences as a hidden attribute to self; also, save it as a csv file.
        if save:
            try:
                print(
                    "Assigning inference dataframe to new attribute self._df_inference."
                )
                # New attribute
                self._df_inference = inference_df
                try:
                    print(f"Saving dataframe as {file_path_csv}")
                    inference_df.to_csv(file_path_csv, index=False)
                except Exception as e:
                    print(
                        f"Error assigning inference dataframe to new attribute self._df_inference: {e}"
                    )
            except Exception as e:
                print(f"Error saving dataframe to csv file: {e}")

        # Return the inference dataframe.
        return inference_df

    def prediction(
        self, lesion_or_image_id: str, filename: str = None
    ) -> Union[None, pd.DataFrame]:
        # If we just want to make a prediction for one or a few lesions/images:
        try:
            if lesion_or_image_id[0] == "I":
                dataframe = self.df[self.df["image_id"] == lesion_or_image_id].copy(
                    deep=True
                )
            elif lesion_or_image_id[0] == "H":
                dataframe = self.df[self.df["lesion_id"] == lesion_or_image_id].copy(
                    deep=True
                )
            else:
                raise ValueError(f"Invalid ID: {lesion_or_image_id}")
        except KeyError as ke:
            raise ValueError(f"ID not found in DataFrame: {lesion_or_image_id}") from ke
        except Exception as e:
            raise ValueError(f"Error processing ID: {e}")

        # Now just call inference on this mini-dataframe:
        output = self.inference(dataframe, filename)

        return output

    def get_hidden_attributes(self) -> Dict[str, Union[Path, str, list, int, float, bool, dict, pd.DataFrame, transforms.Compose, None]]:
        return {
            "source": self.source,
            "model_dir": self.model_dir,
            "data_dir": self.data_dir,
            "filename_stem": self.filename_stem,
            "filename_suffix": self.filename_suffix,
            "_filename": self._filename,
            "overwrite": self.overwrite,
            "restrict_to": self.restrict_to,
            "remove_if": self.remove_if,
            "drop_row_if_missing_value_in": self.drop_row_if_missing_value_in,
            "tvr": self.tvr,
            "seed": self.seed,
            "keep_first": self.keep_first,
            "stratified": self.stratified,                  
            "val_one_img_per_lesion": self.val_one_img_per_lesion,
            "val_expansion_factor": self.val_expansion_factor,
            "to_classify": self.to_classify,
            "sample_size": self.sample_size,
            "label_codes": self.label_codes,
            "label_dict": self.label_dict,
            "num_labels": self.num_labels,             
            "df": self.df,
            "df_combined": self.df_combined,
            "_df_train": self._df_train,
            "_df_val": self._df_val,            
            "_df_inference": self._df_inference,            
            "transform": self.transform,
            "batch_size": self.batch_size,
            "base_learning_rate": self.base_learning_rate,
            
            "code_test": self.code_test,       
        }
    
    def save_attributes_to_file(self):
        # Filter and convert values to strings
        attributes_dict = {
            "data_dir": self.data_dir,            
            "model_dir": self.model_dir,
            "filename_stem": self.filename_stem,
            "filename_suffix": self.filename_suffix,
            "_filename": self._filename,         
            "overwrite": self.overwrite,            
            "restrict_to": self.restrict_to,
            "remove_if": self.remove_if,
            "drop_row_if_missing_value_in": self.drop_row_if_missing_value_in,
            "tvr": self.tvr,
            "seed": self.seed,
            "keep_first": self.keep_first,
            "stratified": self.stratified,            
            "val_one_img_per_lesion": self.val_one_img_per_lesion,
            "val_expansion_factor": self.val_expansion_factor,
            "to_classify": self.to_classify,
            "sample_size": self.sample_size,
            "label_codes": self.label_codes,
            "label_dict": self.label_dict,
            "num_labels": self.num_labels,             
            "transform": self.transform,
            "batch_size": self.batch_size,
            "base_learning_rate": self.base_learning_rate,
            "code_test": self.code_test,       
        }
        filtered_dict = {}
        for key, value in attributes_dict.items():
            if isinstance(value, (str, int, float, bool, list, Path, transforms.Compose, dict)):
                filtered_dict[key] = str(value)
        filepath = self.model_dir.joinpath(self._filename + "_attributes.json")
        # Save the filtered dictionary to a JSON file
        print(f"Attributes saved to file: {filepath}")
        with open(filepath, 'w') as json_file:
            json.dump(filtered_dict, json_file)


'''
END RESNET18 CLASS
'''
    
def aggregate_probabilities(df: pd.DataFrame,
                            method_dict: Union[None, dict] = None,
                            prefix: str = 'prob_',
                            ) -> pd.DataFrame:
    
    output = df.copy()
    
    if method_dict is None:
        method_dict = { 'mean' : output.columns }
    else:
        method_dict['mean'] = [prefix + lesion for lesion in method_dict['mean'] if lesion in output.columns]
        method_dict['max'] = [prefix + lesion for lesion in method_dict['max'] if lesion in output.columns]
        method_dict['min'] = [prefix + lesion for lesion in method_dict['min'] if lesion in output.columns]
    
    mean_probs = output.groupby('lesion_id').mean()
    max_probs = output.groupby('lesion_id').max()
    min_probs = output.groupby('lesion_id').min()

    for col in mean_probs.columns:
        if col in method_dict['mean']:
            output[col] = output['lesion_id'].map(mean_probs[col])
        elif col in method_dict['max']:
            output[col] = output['lesion_id'].map(max_probs[col])
        elif col in method_dict['min']:
            output[col] = output['lesion_id'].map(min_probs[col])
    
    return output        
    
def threshold(probabilities: pd.Series, 
              threshold_dict_help: Union[OrderedDict,None],
              threshold_dict_hinder: Union[OrderedDict,None],
              prefix: str = 'prob_') -> pd.Series:   
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

def get_argmax(row: pd.DataFrame, 
               prefix: str='prob_', 
               threshold_dict_help: Union[OrderedDict,None] = None,
               threshold_dict_hinder: Union[OrderedDict,None] = None,
               inverse_label_codes=Union[dict,None]) -> Union[int, str]:
    # Filter columns based on the prefix
    prob_columns = [col for col in row.index if col.startswith(prefix)] #? why .index and not .columns? No, after applying .groupby, the grroupby column becomes the index of the resulting dataframe or something
    probabilities = row[prob_columns].astype(float)
    
    probabilities = threshold(probabilities=probabilities, 
                              threshold_dict_help=threshold_dict_help,
                              threshold_dict_hinder=threshold_dict_hinder,
                              prefix=prefix,)
        
    max_column = probabilities.idxmax()
    dx = max_column.split('_')[1]  # Split the string and return the second part (after the prefix)
    if inverse_label_codes:
        return inverse_label_codes[dx]  # Return the label if inverse_label_codes is provided
    else:
        return dx  # Otherwise, return the code itself
    
def append_prediction(original_df: pd.DataFrame,
                      probabilities_df: pd.DataFrame, 
                      threshold_dict_help: Union[OrderedDict, None] = None,
                      threshold_dict_hinder: Union[OrderedDict, None] = None,
                      inverse_label_codes: Union[dict,None] = None,
                      prefix: str = 'prob_',) -> pd.DataFrame:    
    
    # Make a copy of the original dataframe
    output_df = original_df.copy()
    
    # Apply the 'get_argmax' function to the probabilities dataframe and append the result to the original
    output_df['pred'] = probabilities_df.apply(get_argmax, 
                                               prefix=prefix, 
                                               threshold_dict_help=threshold_dict_help,
                                               threshold_dict_hinder=threshold_dict_hinder,
                                               inverse_label_codes=inverse_label_codes, 
                                               axis=1)
    return output_df

def df_with_probabilities(model_or_path: Union[resnet18, Path],) -> pd.DataFrame:
    if isinstance(model_or_path, resnet18):
        instance = model_or_path
        try:
            if isinstance(instance._df_inference, pd.DataFrame):
                return instance._df_inference
            else:
                try:
                    filename_csv = instance._filename + "_infer.csv"
                    file_path_csv = instance.model_dir.joinpath(filename_csv)
                    instance._df_inference = pd.read_csv(file_path_csv)
                    return instance._df_inference
                except Exception as e:
                    print(f"Error reading csv file {file_path_csv}: {e}")            
                    return
        except AttributeError as e:
            print(f"{e}")          
            try:
                filename_csv = instance._filename + "_infer.csv"
                file_path_csv = instance.model_dir.joinpath(filename_csv)
                instance._df_inference = pd.read_csv(file_path_csv)
                return instance._df_inference
            except Exception as e:
                print(f"Error reading csv file {file_path_csv}: {e}")            
                return
    elif isinstance(model_or_path, Path):
        file_path_csv = model_or_path
        try:
            output_df = pd.read_csv(file_path_csv)
        except Exception as e:
            print(f"Error reading csv file {file_path_csv} : {e}")
            return
    else:
        print("First argument must be an instance of a model class of a Path to a csv file.")
        return
    return output_df
    
def mode_with_random(x):
    modes = x.mode()
    if not modes.empty:
        return modes[0]
    else:
        max_count = x.value_counts().max()
        modes = x.value_counts()[x.value_counts() == max_count].index.tolist()
        return np.random.choice(modes)

def predictions_mode(df: pd.DataFrame, 
                     pred_col: str = 'pred',) -> pd.DataFrame:
    mode_df = df.groupby('lesion_id')[pred_col].agg(mode_with_random)
    output = df.merge(mode_df, left_on='lesion_id', right_index=True, suffixes=('', '_mode')).drop('pred', axis=1)
    return output