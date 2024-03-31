import os
from pathlib import Path
import pandas as pd
import numpy as np
from IPython.display import display
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

                
class image_n_label:
    def __init__(self, df: pd.DataFrame, label_codes: dict, data_dir: Path, transform=None) -> (Image, str, str):
        self.df = df[["image_id", "label"]].copy() # self.df = df
        self.label_codes = label_codes
        self.data_dir = data_dir
        self.transform = transform
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
#         label = self.df.loc[idx, "label"]
#         image_id = self.df.loc[idx, "image_id"]
         # One-hot encoding the labels
        label = torch.zeros(len(self.label_codes))
        label[self.df.loc[idx, "label"]] = 1
        image_id = self.df.loc[idx, "image_id"]
            
        if self.transform:
            image = self.transform(image)

        return image, label, image_id
    
class resnet18:
    def __init__(self, 
                 df: pd.DataFrame, 
                 train_set: Union[pd.DataFrame, list, str],
                 val_set: Union[pd.DataFrame, list, str],
                 label_codes: dict,
                 data_dir: Path, 
                 model_dir: Path, 
                 transform: List[Callable], # Requires from typing import List, Callable
                 batch_size: int = 32,
                 epochs: int = 10,
                 base_learning_rate: float = 0.001,
                 filename_stem: str = "rn18mc",
                 filename_suffix: str = "",
                 model = models.resnet18(weights="ResNet18_Weights.DEFAULT"),
                 state_dict: Union[None, Dict[str, torch.Tensor]] = None, # from typing import Dict, Union
                 epoch_losses: dict = None,
                ) -> None:
        
        self.df = df 
        self.train_set = train_set
        self.val_set = val_set
        self.label_codes = label_codes
        # Set up self._df_train (new attribute)---depends on the type of input:
        if isinstance(train_set, pd.DataFrame):
            self._df_train = train_set
        elif isinstance(train_set, list):
            self._df_train = self.df[self.df["set"].isin(train_set)]
        elif isinstance(train_set, str):
            self._df_train = self.df[self.df["set"] == train_set]
        else:
            raise ValueError("train_set must be either a DataFrame, a list (e.g. [\'t1\',\'ta\'], or a string (e.g. \'t1'\).")
        
        # Now the same for self._df_val (new attribute):
        if isinstance(val_set, pd.DataFrame):
            self._df_val = val_set
        elif isinstance(val_set, list):
            self._df_val = self.df[self.df["set"].isin(val_set)]
        elif isinstance(val_set, str):
            self._df_val = self.df[self.df["set"] == val_set]
        else:
            raise ValueError("val_set must be either a DataFrame, a list (e.g. [\'v1\',\'va\'], or a string (e.g. \'v1'\).")            
        
        # Continuing...
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.transform = transform
        self.batch_size = batch_size
        self.epochs = epochs
        self.base_learning_rate = base_learning_rate
        self.filename_stem = filename_stem
        self.filename_suffix = filename_suffix
        self.model = model
        self.state_dict = state_dict
        self.epoch_losses = epoch_losses 
        
        self.construct_filename()
        
    def construct_filename(self) -> None:
        # To construct a string for the filename (for saving)
        try:
            if "ta" in self._df_train["set"].unique():
                tcode = "ta"
            else:
                tcode = "t1"
        except:
            tcode = ""
        if self.filename_suffix == "":
            # So that the chance of over-writing an existing file is about 1/900...
            self.filename_suffix = str(np.random.randint(100, 1000))
        # New attribute
        self._filename = "_".join([self.filename_stem, tcode, str(self.epochs) + "e", self.filename_suffix]) 
        
    def train(self) -> None:
        # Define DataLoader for batch processing
#         training_data = image_n_label(self._df_train,self.data_dir, self.transform)
        training_data = image_n_label(self._df_train, self.label_codes, self.data_dir, self.transform)        
        dataloader = DataLoader(training_data, batch_size = self.batch_size, shuffle = True)

        # Load the ResNet18 model 
        model = self.model
        # Unfreeze all layers for fine-tuning
        for param in model.parameters():
            param.requires_grad = True # All layers unfrozen for fine-tuning right away

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
        loss_dict = {"train_loss" : (-1)*np.ones(self.epochs), "val_loss": (-1)*np.ones(self.epochs)}

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for images, labels, _ in dataloader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs.squeeze(), labels)
#                 loss = criterion(outputs.squeeze(), labels.long())
                # Getting "RuntimeError: size mismatch (got input: [5], target: [1])", but not when training on test batches of size 64/128. 
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            # Calculate validation loss for the epoch
            epoch_loss = running_loss / len(dataloader)
            # Add it to the dictionary
            loss_dict["train_loss"][epoch] = epoch_loss

            # Validation step        

            # Define DataLoader for batch processing for validation set
#             validation_data = image_n_label(self._df_val, self.data_dir, self.transform)
            validation_data = image_n_label(self._df_val, self.label_codes, self.data_dir, self.transform)
            val_dataloader = DataLoader(validation_data, batch_size = self.batch_size, shuffle = False)  # No need to shuffle for validation        

            # Set model to evaluation mode
            model.eval() 
            val_running_loss = 0.0
            with torch.no_grad():  # Disable gradient calculation during validation
                for val_images, val_labels, _ in val_dataloader:
                    val_images, val_labels = val_images.to(device), val_labels.to(device)
                    val_outputs = model(val_images)
#                     val_loss = criterion(val_outputs.squeeze(), val_labels.long())
                    val_loss = criterion(val_outputs.squeeze(), val_labels)
                    val_running_loss += val_loss.item()

                    # Calculate validation loss for the epoch
                    val_epoch_loss = val_running_loss / len(val_dataloader)
                    # Add it to the dictionary
                    loss_dict["val_loss"][epoch] = val_epoch_loss

            print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, Validation Loss: {val_epoch_loss:.4f}")

        # Now save the model
        file_path = self.model_dir.joinpath(self._filename + ".pth")
        print(f"Saving model.state_dict() as {file_path}.")
        torch.save(model.state_dict(), file_path)
        
        print("model.state_dict() can now be accessed through state_dict attribute.") 
        print("Train/val losses can now be accessed through epoch_losses attribute.")
        self.epoch_losses = loss_dict
        self.state_dict = model.state_dict()

    def inference(self, df_infer: pd.DataFrame = None, filename: str = None) -> pd.DataFrame:
        if df_infer is None:
            df_infer = self.df
        # Define DataLoader for batch processing
#         inference_data = image_n_label(df_infer, self.data_dir, self.transform)
        inference_data = image_n_label(df_infer, self.label_codes, self.data_dir, self.transform)
        dataloader = DataLoader(inference_data, batch_size=self.batch_size, shuffle=False)
        model = self.model
        # Load the model
        if self.state_dict is None:
            assert filename is not None, "state_dict attribute is None: provide a filename for loading."                
            try:
                file_path = self.model_dir.joinpath(filename)
                model.load_state_dict(torch.load(file_path))
                try:
                    file_path = self.data_dir.joinpath(filename + ".pth")
                    model.load_state_dict(torch.load(file_path))
                except Exception as e:
                    file_path = self.data_dir.joinpath(filename + ".pth")
                    print(f"Error loading {file_path}: {e}.")                    
            except Exception as e:
                file_path = self.data_dir.joinpath(filename + ".pth")
                print(f"Error loading {file_path}: {e}.")
                    
        # Set the model to evaluation mode
        model.eval()  

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)    
        
        # Use DataParallel for parallel processing (if multiple GPUs are available)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        # Dataframe to store image_id and prediction
        cols = ["image_id"] + ["prob_" + label for label in self.label_codes.values()]
        image_id_prob = pd.DataFrame({col_name : pd.NA for col_name in cols}, index = [0])
        
        softmax = nn.Softmax(dim=1)
        # Iterate through the DataLoader and make predictions
        with torch.no_grad():
            for images, labels, image_ids in dataloader:
                # Make predictions using the model
                outputs = model(images)
                # Apply softmax to get probabilities                
                probabilities = softmax(outputs)
                
                series_dict = { }
                series_dict["image_id"] = pd.Series(image_ids)

                for idx, label in enumerate(self.label_codes.values()):
                    series_dict["prob_" + label] = pd.Series(probabilities[:,idx])
                
                batch_df = pd.DataFrame(series_dict)
                
                image_id_prob = pd.concat([image_id_prob, batch_df], axis=0)  
        
        image_id_prob = image_id_prob.dropna(subset=["image_id"])     
        
        return image_id_prob 
    
    def prediction(self, lesion_or_image_id: str, filename: str = None) -> Union[None, pd.DataFrame]:
        try:
            if lesion_or_image_id[0] == "I":
                dataframe = self.df[self.df["image_id"] == lesion_or_image_id].copy(deep=True)
            elif lesion_or_image_id[0] == "H":
                dataframe = self.df[self.df["lesion_id"] == lesion_or_image_id].copy(deep=True)
            else:
                raise ValueError(f"Invalid ID: {lesion_or_image_id}")
        except KeyError as ke:
            raise ValueError(f"ID not found in DataFrame: {lesion_or_image_id}") from ke
        except Exception as e:
            raise ValueError(f"Error processing ID: {e}")

        output = self.inference(dataframe, filename)

        return output            
    
    def get_hidden_attributes(self) -> Dict[str, Union[str, pd.DataFrame]]:
        return {
            "_df_train": self._df_train,
            "_df_val": self._df_val,
            "_filename": self._filename,
        }    