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

class process:
    def __init__(self, 
                 data_dir: Path, 
                 filename: str, 
                 tvr: int, 
                 seed: int, 
                 keep_first: bool,
                 dxs: list,
                 label_codes: dict = None,) -> None:

        self.data_dir = data_dir
        self.filename = filename
        self.file_path = self.data_dir.joinpath(self.filename)
        self.dxs = dxs
        
        # Load metadata.csv into a dataframe.
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"Successfully loaded file '{self.file_path}'.")
        except Exception as e:
            print(f"Error loading file '{self.file_path}': {e}")
        
        # Insert 'num_images' column to the right of 'lesion_id' column.
        try:
            self.df.insert(1, 'num_images', self.df['lesion_id'].map(self.df['lesion_id'].value_counts()))
            print(f"Inserted \'num_images\' column.")
        except Exception as e:
            print(f"Error inserting \'num_images\' column: {e}")  
        
        # Insert 'class' column to the right of 'dx' column.
        try:
            all_dxs = {"nv", "bkl", "vasc", "df", "mel", "bcc", "akiec"}
            not_dxs = all_dxs - set(self.dxs)
            map_dict = {**{dx : dx for dx in self.dxs} , **{dx : "other" for dx in not_dxs}}
            self.df.insert(4, 'label', self.df['dx'].map(map_dict))
            self.dxs.append("other")
            print(f"Inserted \'label\' column.")            
        except Exception as e:
            print(f"Error inserting \'label\' column: {e}.")
        # Create label_codes dictionary for converting string labels to integers.
        try:
            labels = self.df["label"].unique()
            self.label_codes = { label : index for index, label in enumerate(labels) }
        except Exception as e:
            print(f"Error creating label_codes diciontary: {e}.")
        
        # Add 'set' column indicating which set (train/val) each lesion (and its images) belongs to.
        self.tvr = tvr
        self.seed = seed
        self.keep_first = keep_first
        
        # Collect distinct lesions
        try:
            distinct_lesions = self.df['lesion_id'].unique()
        except Exception as e:
            print(f"Error obtaining array of unique \'lesion_id\'s: {e}")
        # Determine the number of distinct lesions to be represented in the training set
        if self.tvr == 0:
            distinct_lesions_train_size = 0
        else:
            distinct_lesions_train_size = int(distinct_lesions.shape[0]*(1/(1 + 1/self.tvr)))
        
        # Randomly select that many distinct lesions to be represented in our training set. 
        np.random.seed(self.seed)
        t = np.random.choice(distinct_lesions, distinct_lesions_train_size, replace = False)
        # The distinct lesions not represented in our training set will be represented in our validation set.
        v = distinct_lesions[~np.isin(distinct_lesions, t)]        
        
        # For the one-image-per-lesion scenarios
        try:
            if self.keep_first: # Keep first image of each lesion
                t1 = self.df[self.df['lesion_id'].isin(t)].drop_duplicates(subset=['lesion_id'], keep='first')['image_id']   
                v1 = self.df[self.df['lesion_id'].isin(v)].drop_duplicates(subset=['lesion_id'], keep='first')['image_id']   
            else: # Keep a random image of each lesion
                t1 = self.df[self.df['lesion_id'].isin(t)].sample(frac=1, random_state=self.seed).drop_duplicates(subset=['lesion_id'], keep='first')['image_id']         
                v1 = self.df[self.df['lesion_id'].isin(v)].sample(frac=1, random_state=self.seed).drop_duplicates(subset=['lesion_id'], keep='first')['image_id']   
        except Exception as e:
            print(f"Error dropping duplicate \'lesion_id\'s and keeping one \'image_id\' per \'lesion_id\': {e}.")

        # For the all-images scenarios    
        ta = self.df[(self.df['lesion_id'].isin(t)) & ~(self.df['image_id'].isin(t1))]['image_id']
        va = self.df[(self.df['lesion_id'].isin(v)) & ~(self.df['image_id'].isin(v1))]['image_id']

        # Add labels to our dataframe, in a new column called 'set'
        try:
            self.df.loc[self.df['image_id'].isin(t1),'set'] = 't1'
            self.df.loc[self.df['image_id'].isin(v1),'set'] = 'v1'
            self.df.loc[self.df['image_id'].isin(ta),'set'] = 'ta'
            self.df.loc[self.df['image_id'].isin(va),'set'] = 'va'
            print("Added \'set\' column with tags \'t1\', \'v1\', \'ta\', and \'va\'.")
        except Exception as e:
            print(f"Error adding \'set\' column with tags indicating train/val assignment: {e}.")
         
    # DIAGNOSIS DISTRIBUTION FOR LESIONS AND IMAGES, AFTER TRAIN/VAL SPLIT

    def dx_dist(self, subset: str = "all", across: str = "lesions") -> None:
        df = self.df
        # Shorthand for relevant conditions
        T1 = df['set'] == "t1" # train on one image per lesion
        V1 = df['set'] == "v1" # validate on one image per lesion
        TA = T1 | (df['set'] == "ta") # train on all images (recall that this means t1 OR ta)
        VA = V1 | (df['set'] == "va") # validate on all images (recall that this means v1 OR va)

        tot_lesions = df[T1 | V1].shape[0]
        tot_images = df[TA | VA].shape[0]

        if subset == "all":
            subset = "overall"
            if across == "lesions":
                col = df[T1 | V1]["label"]
            else: 
                across == "images"
                col = df[TA | VA]["label"]
        elif subset == "train":
            if across == "lesions":
                col = df[T1]["label"]
            else: 
                across == "images"
                col = df[TA]["label"]
        else:  
            if across == "lesions":
                col = df[V1]["label"]
            else:  
                across == "images"
                col = df[VA]["label"]

        # Print heading
        print("="*45 + f"\nDistribution of {across} by diagnosis: {subset}\n".upper() + "="*45)
        # Get the frequencies and place them in a dataframe
        dx_breakdown_dist = pd.concat([col.value_counts(), col.value_counts(normalize=True).mul(100).round(2)],axis=1, keys=['freq', '%'])
        dx_breakdown_dist.index.names = ['dx']
        # Display the results
        try:
            display(dx_breakdown_dist.T)
        except:
            print(dx_breakdown_dist.T)
        # Print relative sizes of train/val
        if subset == "overall":
            print(f"Total {across}: {col.shape[0]}.\n")
        else:
            if across == "lesions":
                print(f"Total {across}: {col.shape[0]} ({100*col.shape[0]/tot_lesions:.2f}% of all {across}).\n")
            elif across == "images":
                print(f"Total {across}: {col.shape[0]} ({100*col.shape[0]/tot_images:.2f}% of all {across}).\n")
                
                
# FOR THE NEURAL NETWORK

class image_n_label:
    def __init__(self, df: pd.DataFrame, label_codes: dict, subset: list, data_dir: Path, transform=None) -> (Image, str, str):
        self.df = df
        self.label_codes = label_codes
        self.subset = subset
        self.data_dir = data_dir
        self.transform = transform
    
        try:
            self.df = self.df[self.df["set"].isin(subset)]
        except Exception as e:
            print(f"Encountered error while attempted to restrict dataframe to {train}: {e}.")
        self.df.reset_index(inplace=True)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        img_name = self.data_dir.joinpath(self.df.loc[idx, "image_id"] + ".jpg")  
        image = Image.open(img_name)
        label = self.df.loc[idx, "label"]
        label_code = self.label_codes[label]
        image_id = self.df.loc[idx, "image_id"]
            
        if self.transform:
            image = self.transform(image)

        return image, label_code, image_id
    
class resnet18:
    def __init__(self, 
                 df: pd.DataFrame, 
                 train_set: list,
                 dxs: list, 
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
                 state_dict: Union[None, Dict[str, torch.Tensor]] = None, # From typing import Dict, Union
                 epoch_losses: dict = None,
                ) -> None:
        self.df = df
        self.train_set = train_set
        self.dxs = dxs
        self.label_codes = label_codes
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
        
    def construct_filename(self) -> str:
        # To construct a string for the filename (for saving)
        if "ta" in self.train_set:
            tcode = "ta"
        else:
            tcode = "t1"
        if self.filename_suffix == "":
            # So that the chance of over-writing an existing file is about 1/900...
            self.filename_suffix = str(np.random.randint(100, 1000))
        filename = "_".join([self.filename_stem, tcode, str(self.epochs) + "e", self.filename_suffix])   
        return filename
        
    def train(self) -> None:
        # Define DataLoader for batch processing
        train_set = image_n_label(self.df, self.label_codes, self.train_set, self.data_dir, self.transform)        
        dataloader = DataLoader(train_set, batch_size = self.batch_size, shuffle = True)

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
                loss = criterion(outputs.squeeze(), labels.long())
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            # Calculate validation loss for the epoch
            epoch_loss = running_loss / len(dataloader)
            # Add it to the dictionary
            loss_dict["train_loss"][epoch] = epoch_loss

            # Validation step        

            # Define DataLoader for batch processing for validation set
            val_set = image_n_label(self.df, self.label_codes, ["v1"], self.data_dir, self.transform)
            val_dataloader = DataLoader(val_set, batch_size = self.batch_size, shuffle = False)  # No need to shuffle for validation        

            # Set model to evaluation mode
            model.eval() 
            val_running_loss = 0.0
            with torch.no_grad():  # Disable gradient calculation during validation
                for val_images, val_labels, _ in val_dataloader:
                    val_images, val_labels = val_images.to(device), val_labels.to(device)
                    val_outputs = model(val_images)
                    val_loss = criterion(val_outputs.squeeze(), val_labels.long())
                    val_running_loss += val_loss.item()

                    # Calculate validation loss for the epoch
                    val_epoch_loss = val_running_loss / len(val_dataloader)
                    # Add it to the dictionary
                    loss_dict["val_loss"][epoch] = val_epoch_loss

            print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, Validation Loss: {val_epoch_loss:.4f}")

        filename = resnet18.construct_filename(self)
        # Now save the model
        file_path = self.model_dir.joinpath(filename + ".pth")
        print(f"Saving model.state_dict() as {file_path}.")
        torch.save(model.state_dict(), file_path)
        
        print("model.state_dict() can now be accessed through state_dict attribute.") 
        print("Train/val losses can now be accessed through epoch_losses attribute.")
        self.epoch_losses = loss_dict
        self.state_dict = model.state_dict()

    def inference(self, filename: str = None) -> pd.DataFrame:
        # Define DataLoader for batch processing
        infer_set = image_n_label(self.df, self.label_codes, ["t1", "ta", "v1", "va"], self.data_dir, self.transform)
        dataloader = DataLoader(infer_set, batch_size=self.batch_size, shuffle=False)
        model = self.model
        # Load the model
        if self.state_dict is None:
            assert filename is not None, "state_dict attribute is None: provide a filename for loading."                
            try:
                file_path = self.modeel_dir.joinpath(filename)
                model.load_state_dict(torch.load(file_path))
                try:
                    file_path = self.data_dir.joinpath(filename + ".pth")
                    model.load_state_dict(torch.load(file_path))
                except Exception as e:
                    print(f"Error loading {file_path}: {e}.")                    
            except Exception as e:
                    print(f"Error loading {file_path}: {e}.")
                    
        # Set the model to evaluation mode
        model.eval()  

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)    
        
        # Use DataParallel for parallel processing (if multiple GPUs are available)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        # Dataframe to store image_id and prediction
        cols = ["image_id"] + ["prob_" + label for label in self.label_codes.keys()]
        image_id_prob = pd.DataFrame({col_name : pd.NA for col_name in cols}, index = [0])
        
        softmax = nn.Softmax(dim=1)
        # Iterate through the DataLoader and make predictions
        with torch.no_grad():
            for images, labels, image_ids in dataloader:
                # Make predictions using the model
                outputs = model(images)
                # Apply softmax to get probabilities                
                probabilities = softmax(outputs)
#                 probabilities_np = probabilities.cpu().numpy().flatten()
                
                series_dict = { }
                series_dict["image_id"] = pd.Series(image_ids)#, name = "image_id")

                for idx, label in enumerate(self.label_codes.keys()):
                    series_dict["prob_" + label] = pd.Series(probabilities[:,idx])
                
                batch_df = pd.DataFrame(series_dict)
                
                image_id_prob = pd.concat([image_id_prob, batch_df], axis=0)  
        
        image_id_prob = image_id_prob.dropna(subset=["image_id"])     
        
        return image_id_prob                