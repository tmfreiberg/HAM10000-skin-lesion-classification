import os
from pathlib import Path
import pandas as pd
import numpy as np
from IPython.display import display
from typing import Dict, Union


class process:
    def __init__(
        self,
        data_dir: Path,
        csv_filename: str = "metadata.csv",
        tvr: int = 3,
        seed: int = 0,
        keep_first: bool = False,
        stratified: bool = True,
        to_classify: list = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
        sample_size: Union[None, dict] = None,
    ) -> None:

        self.data_dir = data_dir
        self.csv_filename = csv_filename
        # New attribute: self._csv_file_path
        self._csv_file_path: Path = self.data_dir.joinpath(self.csv_filename)
        # Continuing...
        self.tvr = tvr
        self.seed = seed
        self.keep_first = keep_first
        self.stratified = stratified
        self.to_classify = to_classify
        self.sample_size = sample_size
        # Load dataframe
        self.load()
        # Insert 'num_images' column
        self.insert_num_images()
        # Create label_dict
        self.labels_to_index()
        # Insert 'label' column to the right of 'dx' column in self.df
        self.insert_label_column()
        # Insert 'set' column indicating train/val assignment
        self.train_val_split()
        # Balance classes
        if self.sample_size is not None:
            self.balance()
    
    def load(self):
        # Load metadata.csv into a dataframe.
        try:
            self.df = pd.read_csv(self._csv_file_path)
            print(f"Successfully loaded file '{self._csv_file_path}'.")
        except Exception as e:
            print(f"Error loading file '{self._csv_file_path}': {e}")

    def insert_num_images(self) -> None:
        # Insert 'num_images' column to the right of 'lesion_id' column.
        try:
            self.df.insert(
                1,
                "num_images",
                self.df["lesion_id"].map(self.df["lesion_id"].value_counts()),
            )
            print(
                f"Inserted 'num_images' column in dataframe, to the right of 'lesion_id' column."
            )
        except Exception as e:
            print(f"Error inserting 'num_images' column: {e}")

    def labels_to_index(self) -> None:
        # Map string labels ('mel' etc.) to integers (0,1,2,...)
        all_dxs: set = set(
            self.df["dx"].unique()
        )  # {"akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"}
        try:
            care_about = set(self.to_classify).intersection(all_dxs)
            # Maybe there were some classes in self.to_classify that don't actually exist in the dataframe:
            self.to_classify = list(care_about)
            others = all_dxs - care_about
            if others == {}:
                # New attribute
                self._label_dict = {dx: i for i, dx in enumerate(care_about)}
                # New attribute 
                self._label_codes = {i: dx for i, dx in enumerate(care_about)}
            else:
                self._label_dict = {dx : 0 for dx in others}
                self._label_dict.update({dx: i + 1 for i, dx in enumerate(care_about)})
                # New attribute
                self._label_codes = {0 : 'other'}
                self._label_codes.update({i + 1: dx for i, dx in enumerate(care_about)})              
            # New attribute
            self._num_labels = len(self._label_dict)                
            print("Created self._label_dict (maps labels to indices).")
        except Exception as e:
            print(f"Error creating label_dict: {e}")

    def insert_label_column(self) -> None:
        # Insert 'label' column to the right of 'dx' column.
        try:
            self.df.insert(4, "label", self.df["dx"].map(self._label_dict))
            print(f"Inserted 'label' column in dataframe, to the right of 'dx' column.")
        except Exception as e:
            print(f"Error inserting 'label' column: {e}.")

    def train_val_split(self) -> None:
        # Add 'set' column indicating which set (train/val) each lesion (and its images) belongs to.

        # Determine the proportion of distinct lesions (or class of lesions if stratified) to be represented in the training set
        tvr_multiplier = self.tvr / (self.tvr + 1)

        if self.stratified:
            # Create train_size dictionary such that train_size[0] is the number of lesions of class 0 to go intro training set, etc.
            train_size = dict(
                self.df.drop_duplicates(subset=["lesion_id"], keep="first")[
                    "label"
                ].value_counts()
            )
            train_size = {
                key: int(tvr_multiplier * value) for (key, value) in train_size.items()
            }

            # Randomly select that many distinct lesions from each class to be represented in our training set
            np.random.seed(self.seed)

            t = dict.fromkeys(train_size.keys())
            v = dict.fromkeys(train_size.keys())
            t1 = dict.fromkeys(train_size.keys())
            v1 = dict.fromkeys(train_size.keys())
            ta = dict.fromkeys(train_size.keys())
            va = dict.fromkeys(train_size.keys())

            for label, t_size in train_size.items():
                distinct_lesions = self.df[self.df["label"] == label][
                    "lesion_id"
                ].unique()
                t[label] = np.random.choice(distinct_lesions, t_size, replace=False)
                # The distinct lesions not represented in our training set will be represented in our validation set.
                v[label] = distinct_lesions[~np.isin(distinct_lesions, t[label])]

                # The one-image-per-lesion scenario
                if self.keep_first:  # Keep first image of each lesion
                    t1[label] = self.df[
                        self.df["lesion_id"].isin(t[label])
                    ].drop_duplicates(subset=["lesion_id"], keep="first")["image_id"]
                    v1[label] = self.df[
                        self.df["lesion_id"].isin(v[label])
                    ].drop_duplicates(subset=["lesion_id"], keep="first")["image_id"]
                else:  # Keep a random image of each lesion
                    t1[label] = (
                        self.df[self.df["lesion_id"].isin(t[label])]
                        .sample(frac=1, random_state=self.seed)
                        .drop_duplicates(subset=["lesion_id"], keep="first")["image_id"]
                    )
                    v1[label] = (
                        self.df[self.df["lesion_id"].isin(v[label])]
                        .sample(frac=1, random_state=self.seed)
                        .drop_duplicates(subset=["lesion_id"], keep="first")["image_id"]
                    )

                # For the all-images scenarios
                ta[label] = self.df[
                    (self.df["lesion_id"].isin(t[label]))
                    & ~(self.df["image_id"].isin(t1[label]))
                ]["image_id"]
                va[label] = self.df[
                    (self.df["lesion_id"].isin(v[label]))
                    & ~(self.df["image_id"].isin(v1[label]))
                ]["image_id"]
                # Add labels to our dataframe, in a new column called 'set'
                self.df.loc[self.df["image_id"].isin(t1[label]), "set"] = "t1"
                self.df.loc[self.df["image_id"].isin(v1[label]), "set"] = "v1"
                self.df.loc[self.df["image_id"].isin(ta[label]), "set"] = "ta"
                self.df.loc[self.df["image_id"].isin(va[label]), "set"] = "va"

            print(
                f"Added 'set' column to dataframe, with values 't1', 'v1', 'ta', and 'va', to the right of 'localization' column."
            )

        else:
            # Collect distinct lesions
            distinct_lesions = self.df["lesion_id"].unique()

            distinct_lesions_train_size = int(
                tvr_multiplier * distinct_lesions.shape[0]
            )
            # Randomly select that many distinct lesions to be represented in our training set.
            np.random.seed(self.seed)
            t = np.random.choice(
                distinct_lesions, distinct_lesions_train_size, replace=False
            )

            # The distinct lesions not represented in our training set will be represented in our validation set.
            v = distinct_lesions[~np.isin(distinct_lesions, t)]

            # For the one-image-per-lesion scenarios (t1/v1 sets)
            if self.keep_first:  # Keep first image of each lesion
                t1 = self.df[self.df["lesion_id"].isin(t)].drop_duplicates(
                    subset=["lesion_id"], keep="first"
                )["image_id"]
                v1 = self.df[self.df["lesion_id"].isin(v)].drop_duplicates(
                    subset=["lesion_id"], keep="first"
                )["image_id"]
            else:  # Keep a random image of each lesion
                t1 = (
                    self.df[self.df["lesion_id"].isin(t)]
                    .sample(frac=1, random_state=self.seed)
                    .drop_duplicates(subset=["lesion_id"], keep="first")["image_id"]
                )
                v1 = (
                    self.df[self.df["lesion_id"].isin(v)]
                    .sample(frac=1, random_state=self.seed)
                    .drop_duplicates(subset=["lesion_id"], keep="first")["image_id"]
                )

            # For the all-images scenarios
            ta = self.df[
                (self.df["lesion_id"].isin(t)) & ~(self.df["image_id"].isin(t1))
            ]["image_id"]
            va = self.df[
                (self.df["lesion_id"].isin(v)) & ~(self.df["image_id"].isin(v1))
            ]["image_id"]

            self.df.loc[self.df["image_id"].isin(t1), "set"] = "t1"
            self.df.loc[self.df["image_id"].isin(v1), "set"] = "v1"
            self.df.loc[self.df["image_id"].isin(ta), "set"] = "ta"
            self.df.loc[self.df["image_id"].isin(va), "set"] = "va"

            print(
                f"Added 'set' column to dataframe, with values 't1', 'v1', 'ta', and 'va', to the right of 'localization' column."
            )
            
        # We create new attributes df_train1, df_train_a, df_val1, and df_val_a.
        # They are references to df (all point to the same underlying dataframe object in memory).
        # Any modifications made to self.df will also affect self.df_trian1 etc., and vicer-versa.
        self._df_train1 = self.df[self.df["set"] == "t1"]
        self._df_train_a = self.df[(self.df["set"] == "t1") | (self.df["set"] == "ta")]
        self._df_val1 = self.df[self.df["set"] == "v1"]
        self._df_val_a = self.df[(self.df["set"] == "v1") | (self.df["set"] == "va")] 

        # Finally, for convenience, we create a test dataframe (again a reference to self.df).
        # By "test" we mean something to test our code, not a "test set" for our models.
        self._df_sample_batch = self.df.sample(n=64, random_state=self.seed)

    # DIAGNOSIS DISTRIBUTION FOR LESIONS AND IMAGES, AFTER TRAIN/VAL SPLIT

    def dx_dist(self, subset: str = "all", across: str = "lesions") -> None:
        df = self.df
        # Shorthand for relevant conditions
        T1 = df["set"] == "t1"  # train on one image per lesion
        V1 = df["set"] == "v1"  # validate on one image per lesion
        TA = T1 | (
            df["set"] == "ta"
        )  # train on all images (recall that this means t1 OR ta)
        VA = V1 | (
            df["set"] == "va"
        )  # validate on all images (recall that this means v1 OR va)

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
        print(
            "=" * 45
            + f"\nDistribution of {across} by diagnosis: {subset}\n".upper()
            + "=" * 45
        )
        # Get the frequencies and place them in a dataframe
        dx_breakdown_dist = pd.concat(
            [col.value_counts(), col.value_counts(normalize=True).mul(100).round(2)],
            axis=1,
            keys=["freq", "%"],
        )
        
        dx_breakdown_dist.index.names = ["dx"]
        
        try:
            index_mapping = { value : key for key, value in self._label_dict.items() if value != 0 }
            if len({ key : value for key, value in self._label_dict.items() if value == 0}) > 1:
                index_mapping.update({ 0 : 'other'})
            else:
                index_mapping.update({ key : value for key, value in self._label_dict.items() if value == 0})
            dx_breakdown_dist.index = dx_breakdown_dist.index.map(index_mapping)
        except:
            pass
        
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
                print(
                    f"Total {across}: {col.shape[0]} ({100*col.shape[0]/tot_lesions:.2f}% of all {across}).\n"
                )
            elif across == "images":
                print(
                    f"Total {across}: {col.shape[0]} ({100*col.shape[0]/tot_images:.2f}% of all {across}).\n"
                )
                
    def balance(self) -> None:
        print("Balancing classes in training set.")
        df = self._df_train_a.copy()
        sample_image_list = []

        if "other" in self.sample_size.keys():
            for i, dx in self._label_codes.items():
                try: # if dx is not a key of self.sample_size, it will simply be skipped
                    N = self.sample_size[dx]
                    class_df = df[df['label'] == i].copy()
                    D = class_df['lesion_id'].nunique()
                    Q, R = divmod(N,D) 
                    # N = Q*D + R
                    # We want Q images for each of the D distinct lesion_ids, plus a further one image for R distinct lesion_ids.
                    # Consider a lesion_id and suppose we have num_images of it.
                    class_df['q'], class_df['r'] = divmod(Q,class_df['num_images']) 
                    # Q = q*num_images + r. We want q copies of each image corresponding to this lesion_id, plus a further one copy of r of them.
                    x = class_df.apply(lambda row: [row['image_id']] * row['q'], axis=1)
                    # Add these to the list
                    sample_image_list.extend([item for sublist in x for item in sublist])
                    # Now for the r 'leftover' images for each lesion
                    y_df = pd.DataFrame(columns=class_df.columns)
                    y_df = class_df.groupby('lesion_id').apply(lambda group: group.sample(n=group['r'].iloc[0])).reset_index(drop=True)
                    y = y_df['image_id'].tolist()
                    # Add them to the list
                    sample_image_list.extend(y)
                    # And we still one image for those remaining R lesion_ids.
                    distinct_lesion_ids = class_df[class_df['label'] == i]['lesion_id'].drop_duplicates()
                    remainder_lesion_ids = distinct_lesion_ids.sample(n=R, random_state=self.seed, replace=False)
                    merged_df = pd.merge(class_df, remainder_lesion_ids.rename('lesion_id'), on='lesion_id')
                    selected_images = merged_df.groupby('lesion_id')['image_id'].apply(lambda img: np.random.choice(img)).tolist()
                    # Add them to the list
                    sample_image_list.extend(selected_images)
                except:
                    pass
        else:
            for dx, i in self._label_dict.items():
                try: # if dx is not a key of self.sample_size, it will simply be skipped
                    N = self.sample_size[dx]
                    class_df = df[df['dx'] == dx].copy()
                    D = class_df['lesion_id'].nunique()
                    Q, R = divmod(N,D) 
                    # N = Q*D + R
                    # We want Q images for each of the D distinct lesion_ids, plus a further one image for R distinct lesion_ids.
                    # Consider a lesion_id and suppose we have num_images of it.
                    class_df['q'], class_df['r'] = divmod(Q,class_df['num_images']) 
                    # Q = q*num_images + r. We want q copies of each image corresponding to this lesion_id, plus a further one copy of r of them.
                    x = class_df.apply(lambda row: [row['image_id']] * row['q'], axis=1)
                    # Add these to the list
                    sample_image_list.extend([item for sublist in x for item in sublist])
                    # Now for the r 'leftover' images for each lesion
                    y_df = pd.DataFrame(columns=class_df.columns)
                    y_df = class_df.groupby('lesion_id').apply(lambda group: group.sample(n=group['r'].iloc[0])).reset_index(drop=True)
                    y = y_df['image_id'].tolist()
                    # Add them to the list
                    sample_image_list.extend(y)
                    # And we still one image for those remaining R lesion_ids.
                    distinct_lesion_ids = class_df[class_df['label'] == i]['lesion_id'].drop_duplicates()
                    remainder_lesion_ids = distinct_lesion_ids.sample(n=R, random_state=self.seed, replace=False)
                    merged_df = pd.merge(class_df, remainder_lesion_ids.rename('lesion_id'), on='lesion_id')
                    selected_images = merged_df.groupby('lesion_id')['image_id'].apply(lambda img: np.random.choice(img)).tolist()
                    # Add them to the list
                    sample_image_list.extend(selected_images)
                except:
                    pass

        # Group sample_image_list by values and count the occurrences
        sample_image_list_counts = pd.Series(sample_image_list).groupby(pd.Series(sample_image_list)).size().reset_index(name='img_mult')

        # Merge df with sample_image_list_counts based on 'image_id'
        balanced_df = pd.merge(df, sample_image_list_counts, left_on='image_id', right_on='index', how='inner')

        # Expand rows based on 'img_mult' column
        balanced_df = balanced_df.loc[balanced_df.index.repeat(balanced_df['img_mult'])].reset_index(drop=True)

        # Drop the temporary 'index' columns
        balanced_df.drop(['index'], axis=1, inplace=True)

        # Count occurrences of each value in 'lesion_id'
        lesion_id_multiplicity = balanced_df['lesion_id'].value_counts()

        # Insert 'multiplicity' column into DataFrame
        balanced_df.insert(loc=1, column='lesion_mult', value=balanced_df['lesion_id'].map(lesion_id_multiplicity))
        
        # Add these two new columns to self._df_val_a as well, because we are about to merge it with balanced_df.
        # (Just to have everything in one dataframe, for convenience.)
        print("Re-combining balanced training set with validation set self._df_val_a so that everything is in one convenient dataframe: self._df_balanced.")
        val_a = self._df_val_a.copy()
        lesion_id_multiplicity = val_a['lesion_id'].value_counts() # This will be the same as 'num_images' 
        image_id_multiplicity = val_a['image_id'].value_counts() # This will be 1
        val_a.insert(loc=1, column='lesion_mult', value=val_a['lesion_id'].map(lesion_id_multiplicity))
        val_a.insert(loc=len(val_a.columns), column='img_mult', value=val_a['image_id'].map(image_id_multiplicity))
        
        # Now concatenate balanced_df with val_a
        balanced_df = pd.concat([balanced_df, val_a], ignore_index=True)
        
        # Re-order the columns
        new_column_order = list(balanced_df.columns)[:4] + ['img_mult'] + list(balanced_df.columns)[4:-1]
        # New attribute
        self._df_balanced = balanced_df[new_column_order]        
                
    def get_hidden_attributes(self) -> Dict[str, Union[Path, str, dict, int, pd.DataFrame]]:
        return {
            "_csv_file_path": self._csv_file_path,
            "_label_dict": self._label_dict,
            "_label_codes": self._label_codes,
            "_num_labels": self._num_labels,
            "_df_train1": self._df_train1,
            "_df_train_a": self._df_train_a,
            "_df_val1": self._df_val1,
            "_df_val_a": self._df_val_a,
            "_df_sample_batch": self._df_sample_batch,
            "_df_balanced": self._df_balanced,
        }
    