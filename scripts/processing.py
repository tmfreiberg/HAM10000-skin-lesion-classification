import os
from pathlib import Path
import pandas as pd
import numpy as np
# from IPython.display import display
from typing import Dict, Union
from utils import display, print_header


class process:
    def __init__(
        self,
        data_dir: Path,
        csv_filename: str = "metadata.csv",
        restrict_to: Union[dict, None] = None,
        remove_if: Union[dict, None] = None,
        drop_row_if_missing_value_in: Union[list, None] = None,
        tvr: int = 3,
        seed: int = 0,
        keep_first: bool = False,
        stratified: bool = True,
        to_classify: Union[list,dict] = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
        train_one_img_per_lesion: Union[bool, None] = None,
        val_expansion_factor: Union[int, None] = None,
        sample_size: Union[None, dict] = None,
    ) -> None:

        self.data_dir = data_dir
        self.csv_filename = csv_filename
        # New attribute: self._csv_file_path
        self._csv_file_path: Path = self.data_dir.joinpath(self.csv_filename)
        # Continuing...
        self.restrict_to = restrict_to
        self.remove_if = remove_if
        self.drop_row_if_missing_value_in = drop_row_if_missing_value_in
        
        self.tvr = tvr
        self.seed = seed
        self.keep_first = keep_first
        self.stratified = stratified
        self.to_classify = to_classify
        
        self.train_one_img_per_lesion = train_one_img_per_lesion
        if self.train_one_img_per_lesion is None:
            self.train_one_img_per_lesion = False
        
        self.val_expansion_factor = val_expansion_factor
        self.sample_size = sample_size
        
        # Load dataframe
        self.load()
        
        # Restrict (if applicable)
        if self.restrict_to is not None or self.remove_if is not None:
            self.filtration()
            
        if self.drop_row_if_missing_value_in is not None:
            self.drop_missing()
            
        # Insert 'num_images' column
        self.insert_num_images()
        # Create label_dict
        self.labels_to_index()
        # Insert 'label' column to the right of 'dx' column in self.df
        self.insert_label_column()
        # Insert 'set' column indicating train/val assignment
        self.train_val_split()
        
        # Balance classes (if applicable)
        if self.sample_size is not None:
            # New attribute
            self.df_train = self.balance()
            if self.train_one_img_per_lesion:
                print("- Balanced training set (one image per lesion): self.df_train")
            else:
                print("- Balanced training set (all images per lesion): self.df_train")
        else:
            if self.train_one_img_per_lesion:
                self.df_train = self._df_train1
                print("- Training set (not balanced, one image per lesion): self.df_train")
            else:
                self.df_train = self._df_train_a
                print("- Training set (not balanced, all images per lesion): self.df_train")
        
        # Expand validation set (if applicable)
        if self.val_expansion_factor is not None:
            print(f'- Expanding validation set: will combine {self.val_expansion_factor} predictions into one, for each lesion in val set.')
            # New attribute
            self.df_val1 = self.expand_val(one_img_per_lesion = True)
            print("- Expanded validation set (one image per lesion): self.df_val1")
            self.df_val_a = self.expand_val(one_img_per_lesion = False)
            print("- Expanded validation set (use more than one image per lesion): self.df_val_a")
        else:
            # New attribute
            self.df_val1 = self._df_val1
            print("- Validation set (not expanded, one image per lesion): self.df_val1")
            self.df_val_a = self._df_val_a
            print("- Validation set (not expanded, use more than one image per lesion): self.df_val_a")

        # Create a small sample batch dataframe for code testing
        self.sample_batch()

    
    def load(self):
        # Load metadata.csv into a dataframe.
        try:
            self.df = pd.read_csv(self._csv_file_path)
            print(f"- Successfully loaded file '{self._csv_file_path}'.")
        except Exception as e:
            print(f"Error loading file '{self._csv_file_path}': {e}")
            
    def filtration(self) -> None:
        if self.restrict_to is not None:
            self.restrict_to = {
                k: v for k, v in self.restrict_to.items() if k in self.df.columns
            }
            if len(self.restrict_to) > 0:
                print(f"- Removing all records unless:")
                for k, v_list in self.restrict_to.items():
                    print(f"  {k} in {v_list}")
                # Generate the query string based on the self.restrict_to dictionary
                query_str = " & ".join(
                    [f"(`{k}` in {v_list})" for k, v_list in self.restrict_to.items()]
                )
                self.df = self.df.query(query_str)
        if self.remove_if is not None:
            self.remove_if = {
                k: v for k, v in self.remove_if.items() if k in self.df.columns
            }
            if len(self.remove_if) > 0:
                print(f"- Removing all records if:")
                for k, v_list in self.remove_if.items():
                    print(f"  {k} in {v_list}")
                # Generate the query string based on the self.remove_if dictionary
                query_str = " | ".join(
                    [f"~(`{k}` in {v_list})" for k, v_list in self.remove_if.items()]
                )
                self.df = self.df.query(query_str)
                
    def drop_missing(self) -> None:
        # New attribute:
        print(
            "- Dropping rows for which there is a missing value in a column from self.drop_row_if_missing_value_in."
        )
        self.df = self.df.dropna(subset=self.drop_row_if_missing_value_in).copy()            

    def insert_num_images(self) -> None:
        # Insert 'num_images' column to the right of 'lesion_id' column.
        try:
            self.df.insert(
                1,
                "num_images",
                self.df["lesion_id"].map(self.df["lesion_id"].value_counts()),
            )
            print(
                f"- Inserted 'num_images' column in dataframe, to the right of 'lesion_id' column."
            )
        except Exception as e:
            print(f"Error inserting 'num_images' column: {e}")

    def labels_to_index(self) -> None:
        if isinstance(self.to_classify,list):
            # Map string labels ('mel' etc.) to integers (0,1,2,...)
            all_dxs: set = set(
                self.df["dx"].unique()
            )  # {"akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"}
            try:
                care_about = set(self.to_classify).intersection(all_dxs)
                # Maybe there were some classes in self.to_classify that don't actually exist in the dataframe:
                self.to_classify = list(care_about)
                others = all_dxs - care_about
                if others:
                    self.label_dict = {dx : 0 for dx in others}
                    self.label_dict.update({dx: i + 1 for i, dx in enumerate(care_about)})
                    # New attribute
                    self.label_codes = {0 : 'other'}
                    self.label_codes.update({i + 1: dx for i, dx in enumerate(care_about)}) 
                else:
                    # New attribute
                    self.label_dict = {dx: i for i, dx in enumerate(care_about)}
                    # New attribute 
                    self.label_codes = {i: dx for i, dx in enumerate(care_about)}                    
                # New attribute
                self._num_labels = len(self.label_dict)                
            except Exception as e:
                print(f"Error creating self.label_dict: {e}")
        elif isinstance(self.to_classify, dict):
            all_dxs: set = set(self.df["dx"].unique())
            try:
                # Make sure all dx's represented in to_classify actually appear in the data
                for dx_category, dx_list in self.to_classify.items():
                    check_exists = set(dx_list).intersection(all_dxs)
                    self.to_classify[dx_category] = list(check_exists)

                # Remove any dx_category, dx_list item from the dictionary if the dx_list is empty
                self.to_classify = { dx_category : dx_list for dx_category, dx_list in self.to_classify.items() if dx_list }

                self.label_codes = { i + 1 : dx_category for i, dx_category in enumerate(self.to_classify.keys()) }

                care_about = set()
                for dx_list in self.to_classify.values():
                    care_about.update(dx_list)
                others = all_dxs - care_about
                if others:
                    self.to_classify['other'] = list(others)
                    self.label_codes[0] = 'other'
                else:
                    self.label_codes = { j - 1 : dx_category for j, dx_category in self.label_codes.items() }            

                self.label_dict = { }
                for i, dx_category in self.label_codes.items():
                    for dx in self.to_classify[dx_category]:
                        self.label_dict[dx] = i
                # New attribute
                self._num_labels = len(self.label_dict)                
            except Exception as e:
                print(f"Error creating label_dict: {e}")
                

    def insert_label_column(self) -> None:
        # Insert 'label' column to the right of 'dx' column.
        try:
            self.df.insert(4, "label", self.df["dx"].map(self.label_dict))
            print(f"- Inserted 'label' column in dataframe, to the right of 'dx' column: \n  {self.label_dict}")
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
                f"- Added 'set' column to dataframe, with values 't1', 'v1', 'ta', and 'va', to the right of 'localization' column."
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
                "- Added 'set' column to dataframe, with values 't1', 'v1', 'ta', and 'va', to the right of 'localization' column."
            )
         
        print("- Basic, overall dataframe (pre-train/test split): self.df")
        # We create new attributes _df_train1, _df_train_a, _df_val1, and _df_val_a.
        # They are references to df (all point to the same underlying dataframe object in memory).
        # Any modifications made to self.df will also affect self._df_train1 etc., and vice-versa.
        self._df_train1 = self.df[self.df["set"] == "t1"]
        self._df_train_a = self.df[(self.df["set"] == "t1") | (self.df["set"] == "ta")]
        self._df_val1 = self.df[self.df["set"] == "v1"]
        self._df_val_a = self.df[(self.df["set"] == "v1") | (self.df["set"] == "va")] 

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
        print_header(f"Distribution of {across} by diagnosis: {subset}")
        # Get the frequencies and place them in a dataframe
        dx_breakdown_dist = pd.concat(
            [col.value_counts(), col.value_counts(normalize=True).mul(100).round(2)],
            axis=1,
            keys=["freq", "%"],
        )
        
        dx_breakdown_dist.index.names = ["dx"]
        
        try:
            index_mapping = { value : key for key, value in self.label_dict.items() if value != 0 }
            if len({ key : value for key, value in self.label_dict.items() if value == 0}) > 1:
                index_mapping.update({ 0 : 'other'})
            else:
                index_mapping.update({ key : value for key, value in self.label_dict.items() if value == 0})
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
                
    def balance(self) -> pd.DataFrame:
        print("- Balancing classes in training set.")
        if self.train_one_img_per_lesion:
            df = self._df_train1.copy()
            df['num_images'] = 1
        else:
            df = self._df_train_a.copy()
        
        sample_image_list = []
        sample_set = set(self.sample_size.keys())

        if (sample_set.difference({'other'})).issubset(set(self.label_codes.values())) or "other" in sample_set:
            for i, dx in self.label_codes.items():
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
        if sample_set.difference({'other'}).issubset(set(self.label_dict.keys())):
            for dx, i in self.label_dict.items():
                if dx in self.label_codes.values():                     
                    pass # Would have already covered this above
                else:
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
                        sample_set = sample_set - {dx}
                    except:
                        pass
        else:
            print("Balancing did not work: ensure sample_size keys either all diagnoses (e.g. 'mel'), or all diagnoses categories as given in to_classify.keys() (if it is a dictionary). (Exception: can include \'other\', regardless.)")
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
        
        if self.train_one_img_per_lesion:
            # Merge _df_train1['num_images'] into balanced_df based on 'lesion_id'
            merged_df = pd.merge(balanced_df, self._df_train1[['lesion_id', 'num_images']], on='lesion_id', how='left')
            # Update 'num_images' column in balanced_df with values from _df_train1
            balanced_df['num_images'] = merged_df['num_images_y'].fillna(merged_df['num_images_x'])
        
        # Re-order the columns
        new_column_order = list(balanced_df.columns)[:4] + ['img_mult'] + list(balanced_df.columns)[4:-1]
        
        return balanced_df[new_column_order]  
                
    def expand_val(self, one_img_per_lesion: bool) -> pd.DataFrame:        
        m = self.val_expansion_factor
        if one_img_per_lesion:
            df = self._df_val1.copy()
            
            expanded_val_df = df.reindex(df.index.repeat(m)).reset_index(drop=True)
            
            # Count occurrences of each value in 'lesion_id'
            lesion_id_multiplicity = expanded_val_df['lesion_id'].value_counts()
            
            # Insert 'multiplicity' column into DataFrame
            expanded_val_df.insert(loc=1, column='lesion_mult', value=expanded_val_df['lesion_id'].map(lesion_id_multiplicity))
            
            expanded_val_df['img_mult'] = m
            
        else:
            df = self._df_val_a.copy()
            
            sample_image_list = []
            
            df['q'], df['r'] = divmod(m,df['num_images']) 
            
            # m = q*num_images + r. We want q copies of each image corresponding to this lesion_id, plus a further one copy of r of them.
            x = df.apply(lambda row: [row['image_id']] * row['q'], axis=1)
            
            # Add these to the list
            sample_image_list.extend([item for sublist in x for item in sublist])
            
            # Now for the r 'leftover' images for each lesion
            y_df = pd.DataFrame(columns=df.columns)
            y_df = df.groupby('lesion_id').apply(lambda group: group.sample(n=group['r'].iloc[0])).reset_index(drop=True)
            y = y_df['image_id'].tolist()
            
            # Add them to the list
            sample_image_list.extend(y)
            
            # Group sample_image_list by values and count the occurrences
            sample_image_list_counts = pd.Series(sample_image_list).groupby(pd.Series(sample_image_list)).size().reset_index(name='img_mult')

            # Merge df with sample_image_list_counts based on 'image_id'
            expanded_val_df = pd.merge(df, sample_image_list_counts, left_on='image_id', right_on='index', how='inner')

            # Expand rows based on 'img_mult' column
            expanded_val_df = expanded_val_df.loc[expanded_val_df.index.repeat(expanded_val_df['img_mult'])].reset_index(drop=True)

            # Drop the temporary 'index' columns
            expanded_val_df.drop(['index', 'q', 'r'], axis=1, inplace=True)

            # Count occurrences of each value in 'lesion_id'
            lesion_id_multiplicity = expanded_val_df['lesion_id'].value_counts()

            # Insert 'multiplicity' column into DataFrame
            expanded_val_df.insert(loc=1, column='lesion_mult', value=expanded_val_df['lesion_id'].map(lesion_id_multiplicity))
           
        
        # Re-order the columns
        new_column_order = list(expanded_val_df.columns)[:4] + ['img_mult'] + list(expanded_val_df.columns)[4:-1]

        return expanded_val_df[new_column_order]                       
        
    def sample_batch(self):
        print("- Small sample dataframes for code testing:", end=' ')
        # New attribute
        self._df_train_code_test = pd.DataFrame()
        for dx in self.df_train['dx'].unique():
            df = self.df_train[(self.df_train['dx'] == dx) & (self.df_train['set'].isin(["t1","ta"]))]['lesion_id']
            sampled_lesion_ids = np.random.choice(df, size=5, replace=False)
            sampled_df = self.df_train[self.df_train['lesion_id'].isin(sampled_lesion_ids)]
            self._df_train_code_test = pd.concat([self._df_train_code_test, sampled_df], ignore_index=True)
        self._df_val1_code_test = pd.DataFrame()
        for dx in self.df_val1['dx'].unique():
            df = self.df_val1[(self.df_val1['dx'] == dx) & (self.df_val1['set'].isin(["v1","va"]))]['lesion_id']
            sampled_lesion_ids = np.random.choice(df, size=2, replace=False)
            sampled_df = self.df_val1[self.df_val1['lesion_id'].isin(sampled_lesion_ids)]
            self._df_val1_code_test = pd.concat([self._df_val1_code_test, sampled_df], ignore_index=True) 
        self._df_val_a_code_test = pd.DataFrame()
        for dx in self.df_val_a['dx'].unique():
            df = self.df_val_a[(self.df_val_a['dx'] == dx) & (self.df_val_a['set'].isin(["v1","va"]))]['lesion_id']
            sampled_lesion_ids = np.random.choice(df, size=2, replace=False)
            sampled_df = self.df_val_a[self.df_val_a['lesion_id'].isin(sampled_lesion_ids)]
            self._df_val_a_code_test = pd.concat([self._df_val_a_code_test, sampled_df], ignore_index=True)  
        
        print("self._df_train_code_test, self._df_val1_code_test, self._df_val_a_code_test")     
                    
    def get_hidden_attributes(self) -> Dict[str, Union[Path, str, dict, int, pd.DataFrame]]:
        return {
            "_csv_file_path": self._csv_file_path,
            "label_dict": self.label_dict,
            "label_codes": self.label_codes,
            "_num_labels": self._num_labels,
            "_df_train1": self._df_train1,
            "_df_train_a": self._df_train_a,
            "df_train": self.df_train,                      
            "df_val1": self.df_val1,
            "df_val_a": self.df_val_a,
            "_df_train_code_test": self._df_train_code_test,
            "_df_val1_code_test": self._df_val1_code_test,
            "_df_val_a_code_test": self._df_val_a_code_test,
        }
    