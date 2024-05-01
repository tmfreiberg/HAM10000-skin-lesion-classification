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


class cnn:
    def __init__(
        self,
        source: Union[process, pd.DataFrame],        
        model_dir: Path,
        transform: Union[None, transforms.Compose, List[Callable]],   
        train_set: Union[None, pd.DataFrame] = None,
        label_codes: Union[None, dict] = None,
        data_dir: Union[None, Path] = None,
        val1_set: Union[None, pd.DataFrame] = None,
        val_a_set: Union[None, pd.DataFrame] = None,
        batch_size: Union[None, int] = None,
        epochs: Union[None, int] = None,
        base_learning_rate: Union[None, float] = None,
        filename_stem: Union[None, str] = None,
        filename_suffix: Union[None, str] = None,
        overwrite: Union[bool, None] = None,
        code_test: Union[bool, None] = None,        
        Print: Union[bool,None] = None,
        model: Union[None, models.ResNet, models.EfficientNet] = None, 
        unfreeze_all: Union[None, bool] = None,
        unfreeze_last: Union[None, bool] = None,
        state_dict: Union[None, Dict[str, torch.Tensor]] = None,  
        epoch_losses: Union[dict, None] = None,
    ) -> None:

        self.source = source
#         From source (if an instance of process class):
#         source.data_dir
#         source.csv_filename
#         source.restrict_to
#         source.remove_if
#         source.drop_row_if_missing_value_in
#         source.tvr
#         source.seed
#         source.keep_first
#         source.stratified
#         source.to_classify
#         source.train_one_img_per_lesion
#         source.val_expansion_factor
#         source.sample_size
#         source.label_dict
#         source.label_codes
#         source.df
#         source._df_train1
#         source._df_train_a
#         source._df_val1
#         source._df_val_a
#         source.df_train
#         source.df_val1
#         source.df_val_a
#         source._df_train_code_test
#         source._df_val1_code_test 
#         source._df_val_a_code_test                
        self.model_dir = model_dir
        self.transform = transform
        self.train_set = train_set
        self.label_codes = label_codes
        self.data_dir = data_dir
        self.val1_set = val1_set
        self.val_a_set = val_a_set
        self.batch_size = batch_size
        self.epochs = epochs
        self.base_learning_rate = base_learning_rate
        self.filename_stem = filename_stem
        self.filename_suffix = filename_suffix
        self.overwrite = overwrite
        self.code_test = code_test
        self.Print = Print
        self.model = model
        self.unfreeze_all = unfreeze_all
        self.unfreeze_last = unfreeze_last
        self.state_dict = state_dict
        self.epoch_losses = epoch_losses
        
        if self.batch_size is None:
            self.batch_size = 32
        if self.epochs is None:
            self.epochs = 10
        if self.base_learning_rate is None:
            self.base_learning_rate = 1/1000
        if self.filename_stem is None:
            if self.model is not None:
                if isinstance(self.model, models.ResNet):
                    self.filename_stem = "rn18"
                elif isinstance(self.model, models.EfficientNet):
                    self.filename_stem = "effnet"
                else:
                    self.filename_stem = "cnn"
            else:
                self.filename_stem = "cnn"
        if self.filename_suffix is None:
            self.filename_suffix = ""
        if self.overwrite is None:
            self.overwrite = False
        if self.code_test is None:
            self.code_test = False
        if self.Print is None:
            self.Print = False
        if self.model is None:
            self.model = models.resnet18(weights="ResNet18_Weights.DEFAULT")  
        if self.unfreeze_all is None:
            self.unfreeze_all = False
        if self.unfreeze_all:
            self.unfreeze_last = True
        if self.unfreeze_last is None:
            self.unfreeze_last = True
            
        if isinstance(self.source, process):
            # New attributes
            self.df = self.source.df
            self.df_train = self.source.df_train
            self.label_codes = self.source.label_codes
            self.data_dir = self.source.data_dir
            self.df_val1 = self.source.df_val1
            self.df_val_a = self.source.df_val_a
        elif isinstance(self.source, pd.DataFrame):
            self.df = self.source
            if self.train_set is not None:
                if isinstance(self.train_set, pd.DataFrame):
                    self.df_train = self.train_set
                elif isinstance(self.train_set, str):
                    self.df_train = self.df[self.df['set'] == self.train_set]
                elif isinstance(self.train_set, list):
                    self.df_train = self.df[self.df['set'].isin(self.train_set)]                
            if self.val1_set is not None:
                if isinstance(self.val1_set, pd.DataFrame):
                    self.df_val1 = self.val1_set
                elif isinstance(self.val1_set, str):
                    self.df_val1 = self.df[self.df['set'] == self.val1]
                elif isinstance(self.val1_set, list):
                    self.df_val1 = self.df[self.df['set'].isin(self.val1_set)]
            if self.val_a_set is not None:
                if isinstance(self.val_a_set, pd.DataFrame):
                    self.df_val_a = self.val_a_set
                elif isinstance(self.val_a_set, str):
                    self.df_val_a = self.df[self.df['set'] == self.val_a_set]
                elif isinstance(self.val_a_set, list):
                    self.df_val_a = self.df[self.df['set'].isin(self.val_a_set)] 
                    
        if self.code_test:
            if isinstance(self.source, process):
                print_header("Code test mode")
                to_print = ["- self.epochs set to 1",
                            "self.Print set to True",
                            "self.filename_suffix set to \'test\'",
                            "self.overwrite set to True",
                            "self.df_train, self.df_val1, self.df_val_a replaced with a small number of records",
                            "Change code_test attribute to False and re-create/create new cnn instance after testing is done.\n"]
                print("\n- ".join(to_print))
                self.epochs = 1
                self.Print = True
                self.filename_suffix = self.filename_suffix + "_test"
                self.overwrite = True         
                self.df_train = self.source._df_train_code_test
                self.df_val1 = self.source._df_val1_code_test
                self.df_val_a = self.source._df_val_a_code_test
            else:
                self.code_test = False
                print("Error: can only turn on code test mode if self.source is an instance of the process class.")       

        self.construct_filename()        
        self.save_attributes_to_file()

    def construct_filename(self) -> None:
        # To construct a string for the filename (for saving)
        tcode = ""
        try:
            if "ta" in self.df_train["set"].unique():
                tcode += "ta"
            else:
                tcode += "t1"
        except:
            pass
        balance_code = ""
        try:
            if isinstance(self.source, process) and self.source.sample_size is not None:
                balance_code += "bal"
            else:
                pass
        except:
            pass 
        uf = ""
        try:
            if self.unfreeze_all:
                uf += "ufall"
            elif self.unfreeze_last:
                uf += "uflast"
        except:
            pass

        # Initial filename without suffix
        elements = [self.filename_stem]
        if tcode:
            elements.append(tcode)
        if balance_code:
            elements.append(balance_code)
        if uf:
            elements.append(uf)
        elements.append(str(self.epochs) + "e")
        if self.filename_suffix:
            elements.append(self.filename_suffix)
        
        base_filename = "_".join(elements)

        # Find a unique filename by incrementing a counter
        counter = 0
        while not self.overwrite:
            filename = base_filename + f"_{counter:02d}"
            filepath = self.model_dir.joinpath(filename + ".pth")

            # Check if the file already exists
            if not os.path.exists(filepath):
                break  # Unique filename found
            else:
                counter += 1  # Increment counter for next attempt

        # New attribute
        if self.overwrite:            
            self._filename = base_filename + f"_{counter:02d}" 
            print(f"Existing files will be overwritten. \nBase filename: {self._filename}")
        else:
            self._filename = filename
            print(f"New files will be created. \nBase filename: {self._filename}")

    def train(self) -> None:
        # Define DataLoader for batch processing
        training_data = image_n_label(
            self.df_train, self.label_codes, self.data_dir, self.transform, self.Print
        )
        dataloader = DataLoader(training_data, batch_size=self.batch_size, shuffle=True)

        # Load the ResNet18/EfficientNet model
        model = self.model
        
        # Replace the last layer for classification with appropriate number of labels
        if isinstance(model,models.ResNet):
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, len(self.label_codes))
        elif isinstance(model,models.EfficientNet):
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs, len(self.label_codes))
            
        if self.unfreeze_all:
            # Unfreeze all layers for fine-tuning
            for param in model.parameters():
                param.requires_grad = True  # All layers unfrozen for fine-tuning right away
        elif self.unfreeze_last:
            if isinstance(model,models.ResNet):
                # Identify the final convolutional block and the fully connected layers
                final_block = model.layer4  # Assuming ResNet-18 has a 'layer4' attribute for the final convolutional block
                fully_connected = model.fc  # Assuming ResNet-18 has an 'fc' attribute for the fully connected layers
                # Unfreeze the final convolutional block
                for param in final_block.parameters():
                    param.requires_grad = True
                # Unfreeze the fully connected layers
                for param in fully_connected.parameters():
                    param.requires_grad = True
            elif isinstance(model,models.EfficientNet):
                print("Need to update code for EfficientNet regarding unfreezing final layers")
                return


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
            "val1_loss": (-1) * np.ones(self.epochs),
            "val_a_loss": (-1) * np.ones(self.epochs),
        }

        for epoch in range(num_epochs):
            model.train()
            batch_counter = 0
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
                if batch_counter%10 == 0:
                    print(f"Batch: {batch_counter}, running loss: {running_loss}")
                batch_counter += 1

            # Calculate validation loss for the epoch
            epoch_loss = running_loss / len(dataloader)
            # Add it to the dictionary
            loss_dict["train_loss"][epoch] = epoch_loss
            print(f"Epoch {epoch}, train_loss: {epoch_loss}")

            # Validation step
            # Define DataLoader for batch processing for validation set
            validation_data1 = image_n_label(
                self.df_val1, # one image per lesion
                self.label_codes,
                self.data_dir,
                self.transform,
                self.Print,
            )
            val_dataloader1 = DataLoader(
                validation_data1, batch_size=self.batch_size, shuffle=False
            )  # No need to shuffle for validation

            # Set model to evaluation mode
            if self.Print:
                print("Validating (one image per lesion)...")
            model.eval()
            val_running_loss = 0.0
            val_epoch_loss = -1
            with torch.no_grad():  # Disable gradient calculation during validation
                for val_images, val_labels, _ in val_dataloader1:
                    val_images, val_labels = val_images.to(device), val_labels.to(
                        device
                    )
                    val_outputs = model(val_images)
                    val_loss = criterion(val_outputs, val_labels)
                    if self.Print:
                        print(f"outputs.shape: {outputs.shape}")
                        print(f"val1_loss: {val_loss}")
                    val_running_loss += val_loss.item()

                # Calculate validation loss for the epoch
                val1_epoch_loss = val_running_loss / len(val_dataloader1)
                # Add it to the dictionary
                loss_dict["val1_loss"][epoch] = val1_epoch_loss
                print(f"Epoch {epoch}, val1_loss: {val1_epoch_loss}")
                    
            validation_data_a = image_n_label(
                self.df_val_a, # all images per lesion
                self.label_codes,
                self.data_dir,
                self.transform,
                self.Print,
            )
            val_dataloader_a = DataLoader(
                validation_data_a, batch_size=self.batch_size, shuffle=False
            )  # No need to shuffle for validation

            # Model already in evaluation mode (see val1 above)
            if self.Print:
                print("Validating (all images per lesion)...")
            val_running_loss = 0.0
            val_a_epoch_loss = -1
            with torch.no_grad():  # Disable gradient calculation during validation
                for val_images, val_labels, _ in val_dataloader_a:
                    val_images, val_labels = val_images.to(device), val_labels.to(
                        device
                    )
                    val_outputs = model(val_images)
                    val_loss = criterion(val_outputs, val_labels)
                    if self.Print:
                        print(f"outputs.shape: {outputs.shape}")
                        print(f"val_a_loss: {val_loss}")
                    val_running_loss += val_loss.item()

                # Calculate validation loss for the epoch
                val_a_epoch_loss = val_running_loss / len(val_dataloader_a)
                # Add it to the dictionary
                loss_dict["val_a_loss"][epoch] = val_a_epoch_loss   
                print(f"Epoch {epoch}, val_a_loss: {val_a_epoch_loss}")

            print(
                f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, Validation Loss 1: {val1_epoch_loss:.4f}, Validation Loss a: {val_a_epoch_loss:.4f}"
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

    def get_hidden_attributes(self) -> Dict[str, Union[Path, str, list, int, float, bool, dict, pd.DataFrame, transforms.Compose, models.ResNet, models.EfficientNet, None]]:
        attributes_dict = {
        "self.df": self.df,
        "self.df_train": self.df_train,
        "self.label_codes": self.label_codes,            
        "self.df_val1": self.df_val1,
        "self.df_val_a": self.df_val_a, 
        "self.data_dir": self.data_dir,                
        "self.model_dir": self.model_dir,
        "self.transform": self.transform,
        "self.train_set": self.train_set,
        "self.label_codes": self.label_codes,
        "self.data_dir": self.data_dir,
        "self.val1_set": self.val1_set,
        "self.val_a_set": self.val_a_set,
        "self.batch_size": self.batch_size,
        "self.base_learning_rate": self.base_learning_rate,
        "self.filename_stem": self.filename_stem,
        "self.filename_suffix": self.filename_suffix,
        "self.overwrite": self.overwrite,
        "self.code_test": self.code_test,
        "self.Print": self.Print,
        "self.model": self.model,
        "self.state_dict": self.state_dict,
        "self.epoch_losses": self.epoch_losses,
        "self._filename": self._filename,
        }        
        if isinstance(self.source, process):
            source_dict = {
            "self.source.data_dir": self.source.data_dir,
            "self.source.csv_filename": self.source.csv_filename,
            "self.source.restrict_to": self.source.restrict_to,
            "self.source.remove_if": self.source.remove_if,
            "self.source.drop_row_if_missing_value_in": self.source.drop_row_if_missing_value_in,
            "self.source.tvr": self.source.tvr,
            "self.source.seed": self.source.seed,
            "self.source.keep_first": self.source.keep_first,
            "self.source.stratified": self.source.stratified,
            "self.source.to_classify": self.source.to_classify,
            "self.source.train_one_img_per_lesion": self.source.train_one_img_per_lesion,
            "self.source.val_expansion_factor": self.source.val_expansion_factor,
            "self.source.sample_size": self.source.sample_size,
            "self.source.label_dict": self.source.label_dict,
            "self.source.label_codes": self.source.label_codes,
            "self.source.df": self.source.df,
            "self.source._df_train1": self.source._df_train1,
            "self.source._df_train_a": self.source._df_train_a,
            "self.source._df_val1": self.source._df_val1,
            "self.source._df_val_a": self.source._df_val_a,
            "self.source.df_train": self.source.df_train,
            "self.source.df_val1": self.source.df_val1,
            "self.source.df_val_a": self.source.df_val_a,
            "self.source._df_train_code_test": self.source._df_train_code_test,
            "self.source._df_val1_code_test": self.source._df_val1_code_test, 
            "self.source._df_val_a_code_test": self.source._df_val_a_code_test,  
            }                        
            attributes_dict.update(source_dict)

        return attributes_dict
    
    def save_attributes_to_file(self):
        # Filter and convert values to strings
        attributes_dict = self.get_hidden_attributes()
        filtered_dict = {}
        for key, value in attributes_dict.items():
            if isinstance(value, (str, int, float, bool, list, Path, transforms.Compose, dict, models.ResNet, models.EfficientNet)):
                filtered_dict[key] = str(value)
        filepath = self.model_dir.joinpath(self._filename + "_attributes.json")
        # Save the filtered dictionary to a JSON file
        print(f"Attributes saved to file: {filepath}")
        with open(filepath, 'w') as json_file:
            json.dump(filtered_dict, json_file)


'''
END CNN CLASS
'''

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

def load_dict(model_dir: Path, filename: str) -> dict:
    if '.' in filename:
        filename = filename[:filename.rindex('.')]       
    file_path = model_dir.joinpath(filename + ".json")
    
    loaded_dict = {}
    
    if file_path.is_file():
        with open(file_path, "r", encoding="utf-8") as file:
            try:
                data = json.load(file)  # Load the entire JSON content
                loaded_dict.update(data)  # Update the dictionary with loaded data
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in file {file_path}: {e}")
    
    return loaded_dict

# For making trivial predictions
def trivial_prediction(
    y_train: Union[np.ndarray, pd.DataFrame, pd.Series],  # targets from training set
    label_codes: Union[None, dict],
    class_probs: str = "zero_one",  # or 'proportion', or 'uniform'
    pos_label_code: Union[int, None] = None,  # code of desired positive label
    pos_label: Union[
        int, str
    ] = "majority_class",  # or 'minority_class', or a value in label_codes (i.e. original_class_name)
    num_preds: Union[int, None] = None,  # number of trivial predictions to be made
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    # We'll deal with numpy arrays only...
    if isinstance(y_train, (pd.DataFrame, pd.Series)):
        y_train_arr = y_train.values
    else:
        y_train_arr = y_train

    # We assume the target is 1-dimensional
    if np.ravel(y_train_arr).shape[0] != y_train_arr.shape[0]:
        return

    # If class_codes dictionary is not given, we'll re-create it
    if label_codes is None:
        classes = np.unique(y_train_arr)
        try:
            classes.sort()
        except:
            pass
        label_codes = {
            idx: classes_element for idx, classes_element in enumerate(classes)
        }
    inverse_label_codes = { v : k for k, v in label_codes.items() }

    # Create dictionary whose items are of the form cc : f, where cc is a class code and f is its frequency
    # Initialize the dictionary with frequencies set to zero
    value_counts_dict = dict.fromkeys(label_codes.keys(), 0)

    # Produce list of values and list of counts, where values[i] has frequency counts[i]
    values, counts = np.unique(y_train_arr, return_counts=True)

    # For values (class codes) with count (frequency) at least one, update the corresponding dictionary item
    for value, count in zip(values, counts):
        value_counts_dict[value] = count

    # Find the maximum frequency among class code frequencies
    M = max([v for k, v in value_counts_dict.items()])

    # The first class code with this frequency will be called the majority class
    majority_class_code = next(
        (k for k, v in value_counts_dict.items() if v == M), None
    )

    # We now set the prediction_code, i.e. the class code of the trivial prediction we want to make
    if pos_label_code is not None and pos_label_code in label_codes.keys():
        prediction_code = pos_label_code
    elif type(pos_label) == str and "maj" in pos_label:
        pos_label = "majority_class"
        prediction_code = majority_class_code
    elif type(pos_label) == str and "min" in pos_label:
        pos_label = "minority_class"
        m = min([v for k, v in value_counts_dict.items() if v != 0])
        minority_class_code = next(
            (k for k, v in value_counts_dict.items() if v == m), None
        )
        prediction_code = minority_class_code
    else:
        try:
            prediction_code = inverse_label_codes[pos_label]
        except:
            pos_label = "majority_class"
            prediction_code = majority_class_code

    # Produce an array of desired dimensions, with the trivial prediction code
    if num_preds is None:
        num_preds = y_train.shape[0]

    trivial_prediction_code = np.full(num_preds, fill_value=prediction_code)

    # And do the same but with the unencoded prediction
    trivial_prediction = np.full(
        num_preds, fill_value=label_codes[prediction_code]
    )

    # Now for the probabilities
    # Certain choices for class_probs are incompatible with choices for pos_label
    # E.g. class_probs = 'proportion' is incompatible with pos_label = 'minority_class'
    if "p" in class_probs and pos_label == "minority_class":
        class_probs = "zero_one"  # could also be 'uniform'

    # Initialize the probabilities array
    trivial_probabilities = np.zeros((num_preds, len(label_codes)))

    # Fill in the probabilities
    if "z" in class_probs:
        class_probs = "zero_one"
        trivial_probabilities[:, prediction_code] = 1
    elif "p" in class_probs:
        class_probs = "proportion"
        for c, cc in class_codes.items():
            trivial_probabilities[:, cc] = value_counts_dict[cc] / y_train_arr.shape[0]
    else:
        class_probs = "uniform"
        trivial_probabilities += 1 / len(class_codes)

    return trivial_prediction, trivial_prediction_code, trivial_probabilities