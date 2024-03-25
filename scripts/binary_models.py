import os
from pathlib import Path
import pandas as pd
import numpy as np
from IPython.display import display
from PIL import Image

import torch
import torch.nn as nn
import torchvision.models as models
from typing import List, Callable, Dict, Union
from torchvision.transforms import Compose, Resize, ToTensor
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# import matplotlib.pyplot as plt  
    
class process:
    def __init__(self, 
                 data_dir: Path, 
                 filename: str, 
                 tvr: int, 
                 seed: int, 
                 keep_first: bool,
                 dxs: list,
                 threshold: dict,
                 combine: dict,):

        self.data_dir = data_dir
        self.filename = filename
        self.file_path = self.data_dir.joinpath(self.filename)
        self.dxs = dxs
        self.threshold = threshold
        self.combine = combine
        
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
            print("Added \'set\' column with labels \'t1\', \'v1\', \'ta\', and \'va\'.")
        except Exception as e:
            print(f"Error adding \'set\' column with labels indicating train/val assignment: {e}.")
         
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
                col = df[T1 | V1]["dx"]
            else: 
                across == "images"
                col = df[TA | VA]["dx"]
        elif subset == "train":
            if across == "lesions":
                col = df[T1]["dx"]
            else: 
                across == "images"
                col = df[TA]["dx"]
        else:  
            if across == "lesions":
                col = df[V1]["dx"]
            else:  
                across == "images"
                col = df[VA]["dx"]

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
                
    def random(self, dx: str) -> np.ndarray:
        """
        Returns an array of length df.shape[0] where each element is between 0 and 1, and the proportion of elements greater than threshold is approximately equal to the proportion of distinct lesions of type dx.
        """
        df = self.df
        threshold = self.threshold[dx]

        # Define the probabilities for X being 1 and 0
        prob_x_1 = df[df["dx"] == dx]["lesion_id"].nunique() / df["lesion_id"].nunique()
        prob_x_0 = 1 - prob_x_1

        target_size = df.shape[0]

        # Generate random variable X for the entire array
        X_array = np.random.choice([0, 1], size=target_size, p=[prob_x_0, prob_x_1])

        # Generate random variable Y based on X for the entire array
        Y_array = np.where(
            X_array == 1,
            np.random.uniform(threshold, 1, size=target_size),
            np.random.uniform(0, threshold, size=target_size),
        )

        return Y_array

    def classify(
        self,
        dx: str,
        probabilities: np.ndarray,
        targets: np.ndarray,        
        Print: bool = False,
    ) -> pd.DataFrame:
        """
        Adds three columns to input dataframe: 'pred1' 'prob_' + combine_method, and 'preda'.
        pred1 shows tp, fp, tn, fn based on probabilities from one image per lesion (the image with 'set' label 'v1'), and threshold.
        proba gives a probability that a given lesion is positive (belongs to class dx), depending on the probabilities associated with all images of the lesion, and the combine_method.
        preda shows tp, fp, tn, fn based on proba.
        """
        df = self.df
        threshold = self.threshold[dx]
        combine_method = self.combine[dx]
        
        # Make a copy of df that will be added to and finally returned
        output = df.copy(deep=True)

        # Add columns for targets (ground truth) and probabilities (given by whatever the model in question is)
        output["prob"] = probabilities        
        output["targ"] = targets

        # If probability > threshold, we assign a positive/true/1 label, else negative/false/0 label.
        # Label is tp/tn/fn/fp (true positive etc.) depending on the target and the label (based on probability and threshold).
        # We add a column showing tp/tn/fn/fp.
        P = output["targ"] == True  # target is positive
        N = output["targ"] == False  # target is negative

        PP1 = output["prob"] >= threshold  # Predicted positive label
        PN1 = output["prob"] < threshold  # Predicted negative label

        TP1 = P & PP1  # Predicted label is true positive
        TN1 = N & PN1  # Predicted label is true negative
        FN1 = P & PN1  # Predicted label is false negative
        FP1 = N & PP1  # Predicted label is false positive

        # Now add the column "pred1" (these are predictions about lesions that based on one image only per lesion).
        output["pred1"] = np.select(
            [TP1, TN1, FN1, FP1], ["tp", "tn", "fn", "fp"], default=np.nan
        )

        if combine_method == "min":
            temp_df = output.groupby("lesion_id").min("prob")[["prob"]]
            temp_df.reset_index(inplace=True)
            output = pd.merge(output, temp_df, on="lesion_id", suffixes=("", "_min"))
        elif combine_method == "mean":
            temp_df = output.groupby("lesion_id").mean("prob")[["prob"]]
            temp_df.reset_index(inplace=True)
            output = pd.merge(output, temp_df, on="lesion_id", suffixes=("", "_mean"))
        elif combine_method == "max":
            temp_df = output.groupby("lesion_id").max("prob")[["prob"]]
            temp_df.reset_index(inplace=True)
            output = pd.merge(output, temp_df, on="lesion_id", suffixes=("", "_max"))

        # We now classify as we did with the one-image-per-lesion version.
        # We re-define P and N as indices may have changed since the above merge.
        # We should also re-define PP1, PN1, TP1 etc., but we won't use them again in this function.
        P = output["targ"] == True  # target is positive
        N = output["targ"] == False  # target is negative

        proba = "_".join(["prob", combine_method])
        PPa = output[proba] >= threshold
        PNa = output[proba] < threshold

        TPa = P & PPa
        TNa = N & PNa
        FNa = P & PNa
        FPa = N & PPa

        output["preda"] = np.select(
            [TPa, TNa, FNa, FPa], ["tp", "tn", "fn", "fp"], default=np.nan
        )

        if Print:
            print(
                f"pred1 column is positive (tp/fp) if prob column greater than {threshold}, negative (tn/fn) otherwise."
            )
            print(
                f"preda column is positive (tp/fp) if {proba} column greater than {threshold}, negative (tn/fn) otherwise."
            )
            print(
                f"{proba} is a probability obtained for each lesion by combining probabilities from all of its images."
            )
            try:
                display(output.head())
            except:
                print(output.head())
        return output                
           
    def setup_dictionary(
        self,
        initial_models: list = ["random", "trivial"],
        Print: bool = True,
    ) -> dict:
        """
        Output will be "binary", a dictionary whose keys are the lesion classes/diagnoses dx for dx in dxs.
        binary[dx] will itself be a dictionary that stores information about various models' predictions for the classification task dx/not dx.
        """        
        df = self.df
        dxs = self.dxs
        threshold = self.threshold

        binary = dict.fromkeys(dxs)

        for dx in dxs:
            binary[dx] = {}
            # We have an array of targets (each element is 1/True/positive if image is of a lesion that belongs to class dx, 0/False/negative otherwise).
            binary[dx]["target"] = df["dx"] == dx
            # Each model will assign a probability to each image (we will elsewhere combine probabilities for multiple images into a single probability for the lesion they represent).
            # We can have a random model, a trivial model (all labels positive), a first model trained on one image per lesion (suffix "_t1"), and a first model trained on all images (suffix "_ta").
            binary[dx]["prob"] = dict.fromkeys(initial_models)
            # We can of course add probabilities for as many models as we please, elsewhere.

            # We add a dictionary item with key "df" (dataframe).
            binary[dx]["df"] = dict.fromkeys(initial_models)

            # We can produce "results" for "random" and "trivial" models:
            if "random" in initial_models:
                binary[dx]["prob"]["random"] = process.random(self, dx)
                binary[dx]["df"]["random"] = process.classify(
                    self,
                    dx,
                    binary[dx]["prob"]["random"],
                    binary[dx]["target"],                    
                    Print=False,
                )
            if "trivial" in initial_models:
                binary[dx]["prob"]["trivial"] = np.ones(binary[dx]["target"].shape[0])
                binary[dx]["df"]["trivial"] = process.classify(
                    self,
                    dx,
                    binary[dx]["prob"]["trivial"],
                    binary[dx]["target"],                    
                    Print=False,
                )
            # For other models, we will elsewhere produce a dataframe with results from that model.
        if Print:
            print(
                'Example: suppose you\'ve named this dictionary "binary", and you\'re interested in lesions of type "mel".'
            )
            print(
                'binary["mel"]["target"]: array of ground-truth labels (True/False) for melanoma.'
            )
            print(
                'binary["mel"]["prob"]["trivial"]: array of probabilities corresponding to trivial predictions (all 1\'s).'
            )
            print(
                'binary["mel"]["prob"]["random"]: array of probabilities corresponding to random prediction.'
            )
            print(
                'binary["mel"]["prob"]["your_model"]: would be an array of probabilities corresponding to predictions from "your_model".'
            )
            print(
                'binary["mel"]["df"]["trivial"]: a dataframe showing targets, predictions, and classification. Threshold is given by threshold attribute, so re-run setup_dictionary after updating threshold.'
            )
        return binary


class evaluation:
    @staticmethod
    def metrics(
        preds: np.ndarray, targs: np.ndarray, BETA: list = [1 / 2, 1, 2]
    ) -> dict:
        """
        INPUTS
        - preds: array of shape (n,) or (n,1). Elements are all 1/0 (or equivalent, e.g. True/False).
        - targs: array of shape (n,) or (n,1). Elements are all 1/0 (or equivalent, e.g. True/False).
        - BETA: list of floats

        OUTPUT
        - output:  dictionary with keys 'P', 'N', etc., as below, and
          values equal to number of 1's (or equivalent) in target, etc.
        """
        keys = [
            "sensitivity",
            "precision",
            "F",
            "specificity",
            "BACC",
            "MCC",
            "accuracy",
            "P",
            "N",
            "PP",
            "PN",
            "tp",
            "tn",
            "fp",
            "fn",
        ]
        output = dict.fromkeys(keys)
        fbeta = dict.fromkeys(BETA)
        output["F"] = fbeta
        code = 2 * preds + targs
        # tp: pred = 1 and targ = 1, code = 2*1 + 1 = 3
        # fp: pred = 1 and targ = 0, code = 2*1 + 0 = 2
        # fn: pred = 0 and targ = 1, code = 2*0 + 1 = 1
        # tn: pred = 0 and targ = 0, code = 2*0 + 0 = 0
        tp = int(np.sum(code == 3))
        tn = int(np.sum(code == 0))
        fp = int(np.sum(code == 2))
        fn = int(np.sum(code == 1))
        output["P"] = tp + fn
        output["N"] = tn + fp
        output["PP"] = tp + fp
        output["PN"] = tn + fn
        output["tp"] = tp
        output["tn"] = tn
        output["fp"] = fp
        output["fn"] = fn
        if tp + fn != 0:
            output["sensitivity"] = tp / (tp + fn)
        if tp + fp != 0:
            output["precision"] = tp / (tp + fp)
        if tn + fp != 0:
            output["specificity"] = tn / (tn + fp)
        if tp + fn != 0 and tp + fp != 0:
            sensitivity = output["sensitivity"]
            precision = output["precision"]
            if tp != 0:
                for beta in BETA:
                    fbeta = (beta**2 + 1) / ((beta**2 / sensitivity) + (1 / precision))
                    output["F"][beta] = fbeta
        if tp + fn != 0 and tn + fp != 0:
            sensitivity = output["sensitivity"]
            specificity = output["specificity"]
            output["BACC"] = (sensitivity + specificity) / 2
        # MCC
        numerator = tp * tn - fp * fn
        denominator_terms = np.array([tp + fp, tp + fn, tn + fp, tn + fn]).astype(
            np.float64
        )
        if np.sum(denominator_terms == 0) == 0:
            denominator = np.sqrt(np.prod(denominator_terms))
            output["MCC"] = numerator / denominator
        elif np.sum(denominator_terms == 0) == 1:
            output["MCC"] = numerator

        # Accuracy (irrelevant, and this will show us why).
        if tp + tn + fp + fn != 0:
            output["accuracy"] = (tp + tn) / (tp + tn + fp + fn)
        return output
    
    @staticmethod
    def metrics_table(df_dict: dict, threshold: float = 0.5) -> pd.DataFrame:
        metrics_dx = {}
        for model in df_dict.keys():
            if df_dict[model] is None:
                pass
            else:
                try:
                    df = df_dict[model]
                    targs = df[df["set"] == "v1"]["targ"]
                    preds1 = df[df["set"] == "v1"]["prob"] > threshold
                    a = df.filter(like='prob_', axis=1).columns[0][5:]
                    proba = "prob_" + a
                    predsa = df[df["set"] == "v1"][proba] > threshold              
                except Exception as e:
                    print(f"Problem with {model} dataframe. Have you run process.classify for this model? {e}")
                if model == "trivial":
                    labels = [model]
                else:
                    labels = [model + "_1", model + "_" + a]
                for pred, label in zip([preds1, predsa], labels):
                    metrics_dx[label] = {}
                    for key, value in evaluation.metrics(pred, targs).items():
                        if key != "F":
                            metrics_dx[label][key] = value
                        else:
                            for key, value in evaluation.metrics(pred, targs)["F"].items():
                                metrics_dx[label]["F" + str(key)] = value

                    metrics_dx[label + "_df"] = pd.DataFrame(list(metrics_dx[label].items()), columns = ["metric", label])

            dataframes = { key : value for key, value in metrics_dx.items() if key.endswith("_df")}
            big_df = pd.concat([df.set_index("metric") for df in dataframes.values()], axis=1).astype(object)
        return big_df
    

# FOR THE NEURAL NETWORK

class image_n_label:
    def __init__(self, df: pd.DataFrame, subset: list, dx: str, data_dir: Path, transform=None) -> (Image, str, str):
        self.df = df
        self.subset = subset
        self.dx = dx
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
        label = self.df.loc[idx, "dx"] == self.dx
        image_id = self.df.loc[idx, "image_id"]
        
        if self.transform:
            image = self.transform(image)

        return image, label, image_id
    
class resnet18:
    def __init__(self, 
                 df: pd.DataFrame, 
                 train_set: list,
                 dx: str, 
                 data_dir: Path, 
                 model_dir: Path, 
                 transform: List[Callable], # Requires from typing import List, Callable
                 batch_size: int = 32,
                 epochs: int = 10,
                 base_learning_rate: float = 0.001,
                 filename_stem: str = "rn18",
                 filename_suffix: str = "",
                 model = models.resnet18(weights="ResNet18_Weights.DEFAULT"),
                 state_dict: Union[None, Dict[str, torch.Tensor]] = None, # From typing import Dict, Union
                 epoch_losses: dict = None,
                ) -> None:
        self.df = df
        self.train_set = train_set
        self.dx = dx
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
        filename = "_".join([self.dx, self.filename_stem, tcode, str(self.epochs) + "e", self.filename_suffix])   
        return filename
        
    def train(self) -> None:
        # Define DataLoader for batch processing
        train_set = image_n_label(self.df, self.train_set, self.dx, self.data_dir, self.transform)        
        dataloader = DataLoader(train_set, batch_size = self.batch_size, shuffle = True)

        # Load the ResNet18 model and fine-tune all layers
        model = self.model
        for param in model.parameters():
            param.requires_grad = True # All layers unfrozen for fine-tuning right away

        # Modify the last layer for our binary classification task
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 1)  # Binary classification, so output size is 1

        # Define loss function and optimizer
        criterion = nn.BCEWithLogitsLoss()
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
                loss = criterion(outputs.squeeze(), labels.float())
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            # Calculate validation loss for the epoch
            epoch_loss = running_loss / len(dataloader)
            # Add it to the dictionary
            loss_dict["train_loss"][epoch] = epoch_loss

            # Validation step        

            # Define DataLoader for batch processing for validation set
            val_set = image_n_label(self.df, ["v1"], self.dx, self.data_dir, self.transform)
            val_dataloader = DataLoader(val_set, batch_size = self.batch_size, shuffle = False)  # No need to shuffle for validation        

            # Set model to evaluation mode
            model.eval() 
            val_running_loss = 0.0
            with torch.no_grad():  # Disable gradient calculation during validation
                for val_images, val_labels, _ in val_dataloader:
                    val_images, val_labels = val_images.to(device), val_labels.to(device)
                    val_outputs = model(val_images)
                    val_loss = criterion(val_outputs.squeeze(), val_labels.float())
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
        infer_set = image_n_label(self.df, ["t1", "ta", "v1", "va"], self.dx, self.data_dir, self.transform)
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
#         image_id_prob = pd.DataFrame(columns = ["image_id", "prob"])
        image_id_prob = pd.DataFrame({"image_id": ["x"], "prob": [-1]})
        
        # Iterate through the DataLoader and make predictions
        with torch.no_grad():
            for images, targets, image_ids in dataloader:
                # Make predictions using the model
                outputs = model(images)
                predictions = torch.sigmoid(outputs.squeeze())  # Assuming binary classification

                # Get this batch's image_ids from the dataloader's dataset (possible because of the way we set up our image_n_label function)
                batch_image_ids = pd.Series(image_ids, name = "image_id")
                # Pair each image_id with its corresponding prediction (probability) and put them in our dataframe
                batch_probs = pd.Series(predictions.cpu().numpy(), name = "prob")
                batch = pd.concat([batch_image_ids, batch_probs], axis = 1,  keys=["image_id", "prob"])
                image_id_prob = pd.concat([image_id_prob, batch], axis=0)  
      
        image_id_prob = image_id_prob.drop(image_id_prob[(image_id_prob["image_id"] == "x") & (image_id_prob["prob"] == -1.)].index)
        
        return image_id_prob
# ###########
