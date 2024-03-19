import os
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from typing import Union

# SETTING PATHS IN A PLATFORM-INDEPENDENT WAY

class path_setup:
    default_env_var = "SKIN_LESION_CLASSIFICATION"

    @staticmethod
    def how(path_name: str, osname: str):
        if osname == "nt":
            instructions = [
                "Search for 'Environment Variables' in the Windows search bar.",
                "Click on 'Edit the system environment variables'.",
                "In the System Properties window, click on the 'Environment Variables' button.",
                "Under 'System variables', click on 'New'.",
                f"Enter {path_name} as the variable name and the desired path as the variable value.",
                "Click 'OK' to save the environment variable.",
                "Close all windows and restart the command prompt for the changes to take effect.",
            ]
        else:
            instructions = [
                "Open a terminal window.",
                "Use the following command to set the environment variable:",
                f"    export {path_name}=/path/to/your/directory",
                "Verify that the environment variable has been set by running:",
                f"    echo ${path_name}",
            ]
        print("")
        for n, step in enumerate(instructions):
            print(f"{n + 1}.", step)

    @staticmethod
    def check(project_path_name: str):
        project_path_value = os.getenv(project_path_name)
        if project_path_value is None:
            print(f"\nThe environment variable '{project_path_name}' does not exist.")
            return path_setup.how(project_path_name, os.name)
        elif not os.path.isdir(project_path_value):
            print(
                f"\nThe environment variable '{project_path_name}' does not point to a valid directory path."
            )
            return path_setup.how(project_path_name, os.name)
        else:
            print(
                f"\nThe environment variable '{project_path_name}' contains a valid directory path.\n"
            )
            project_path = Path(project_path_value)
            return print("=" * 3 + f"\nproject_path = {project_path}\n" + "=" * 3)

    @staticmethod
    def change(project_path_name: str):
        new_project_path_name = input("\nEnter new path name, or 'c' to cancel: ")
        if new_project_path_name.lower() == "c":
            return path_setup.check(project_path_name)
        else:
            return path_setup.confirm(new_project_path_name)

    @staticmethod
    def confirm(project_path_name: str = default_env_var):
        yes_or_no: str = ""
        loop_count: int = 0
        while yes_or_no not in ["y", "n", "c"] and loop_count < 3:
            if loop_count == 0:
                print(
                    f"\n'{project_path_name}' is your desired name for the environment variable pointing to the root directory of this project."
                )
                yes_or_no = input(
                    "\nPlease confirm: Do you wish to use this name? (y/n): "
                )
                loop_count += 1
            else:
                yes_or_no = input(
                    f"\nInvalid input. Enter 'y' or 'n'. Otherwise, enter 'c' to cancel. "
                )
                loop_count += 1
        if yes_or_no.lower() == "n":
            return path_setup.change(project_path_name)
        return path_setup.check(project_path_name)

    @staticmethod
    def subfolders(
        base_path: Path, base_name: str = "project", Print: bool = True
    ) -> dict:
        folders = [
            f
            for f in os.listdir(base_path)
            if f[0] != "." and os.path.isdir(base_path.joinpath(f))
        ]
        path = dict.fromkeys(folders)
        path[base_name] = base_path
        if Print:
            print(f"path['{base_name}'] : {path[base_name]}")
        for folder in folders:
            path[folder] = base_path.joinpath(folder)
            if Print:
                print(f"path['{folder}'] : {path[folder]}")
        return path

# LOADING DATA

def process_metadata_csv(csv_path: Union[Path, None] = None) -> pd.DataFrame:
    """
    Process metadata CSV file to add a column for the number of images per lesion.

    Args:
    csv_path (str or Path, optional): Path to the metadata CSV file.
        Defaults to path["images"].

    Returns:
    pd.DataFrame: Processed metadata DataFrame.
    """
    if csv_path is None:
        csv_path = Path(os.getenv("SKIN_LESION_CLASSIFICATION")).joinpath("images")
#         csv_path = path["images"].joinpath("metadata.csv")

    metadata = pd.read_csv(csv_path)
    metadata_duplicates = metadata[metadata.duplicated(subset='lesion_id')].sort_values('lesion_id')
    metadata.insert(1, 'no_of_images', metadata['lesion_id'].map(metadata['lesion_id'].value_counts()))
    return metadata
    
# VIEWING IMAGES    

class view_images:
    def __init__(self):
        pass

    @staticmethod
    def caption_pred(
        image_id: str,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        dxs: list = ["nv", "bkl", "mel", "bcc", "akiec", "vasc", "df"],
    ) -> str:
        pred = "~predictions~\n"
        for dx in dxs:
            pred += dx + ": "
        try:
            prob1 = df1[df1["image_id"] == image_id]["prob_" + dx].values[0]
            pred += f"{100*prob1:.2f}%"
        except:
            pred += "▯"
        pred += " ¦ "
        try:
            prob2 = df2[df2["image_id"] == image_id]["prob_" + dx].values[0]
            pred += f"{100*prob2:.2f}%"
        except:
            pred += "▯"
        pred += "\n"
        return pred

    @staticmethod
    def caption_part(image_id: str, df1: pd.DataFrame, df2: pd.DataFrame) -> str:
        part = "Set: "
        if df1[df1["image_id"] == image_id]["partition"].isnull().values[0]:
            part += "▯"
        else:
            partition1 = df1[df1["image_id"] == image_id]["partition"].values[0]
            part += f"{partition1}"
        part += " ¦ "
        if df2[df2["image_id"] == image_id]["partition"].isnull().values[0]:
            part += "▯"
        else:
            partition2 = df2[df2["image_id"] == image_id]["partition"].values[0]
            part += f"{partition2}"
        return part

    @staticmethod
    def caption_info(image_id: str, df: pd.DataFrame) -> str:
        caption = ""
        row = df["image_id"] == image_id
        cols = [
            "lesion_id",
            "num_images",
            "image_id",
            "dx",
            "dx_type",
            "age",
            "sex",
            "localization",
        ]
        for col in cols:
            try:
                caption += f"{col}: {df.loc[row, col].values[0]}\n"
            except:
                pass
        return caption

    @staticmethod
    def one_lesion_per_row(
        lesion_ids: np.ndarray,
        df: pd.DataFrame,
        path: Path,
        df1: pd.DataFrame = None,
        df2: pd.DataFrame = None,
        maxrows: int = 10,
        maxcols: int = 6,
        partitions: bool = False,
        predictions: bool = False,
    ) -> None:
        if predictions:
            info = [
                "Below each image are probabilities given by our binary classification models.",
                "To the left of ¦ is the probability according to a model trained on exactly one image per lesion in our training set.",
                "To the right of ¦ is the probability according to a model trained on all images of each lesion in our training set.",
                "For instance, 'mel: 10% ¦ 15%' means that our mel/not binary classifiers determined that the probability that the lesion is melanoma is 10% (one image/lesion) and 15% (multiple images/lesion).",
                "\nNote that we're viewing original images here, not the augmented images the models were trained on.\n",
            ]
            for line in info:
                print(line)
        # Remove possible duplicate lesion ids
        lesion_ids = np.unique(lesion_ids)
        # Get all image ids corresponding to each given lesion id.
        image_ids = {
            lesion_id: df[df["lesion_id"] == lesion_id]["image_id"].values
            for lesion_id in lesion_ids
        }
        # Will display grid of images.
        # Number of rows will be number of lesion ids, unless that number exceeds maxrows...
        nrows = min(maxrows, lesion_ids.shape[0])
        # Number of columns will be maximum number of images corresponding to any lesion in our list, unless this exceeds maxcols...
        ncols = min(
            max([image_ids[lesion_id].shape[0] for lesion_id in lesion_ids]), maxcols
        )
        # If there are more lesion ids than can fit in our grid (limited by maxrows), select a random subset of them.
        rand_lesion_ids = np.random.choice(lesion_ids, nrows, replace=False)
        # Similarly, if, for some lesion, we have more images than our maxcols allows, select a random subset of them.
        rand_image_ids = {
            lesion_id: np.random.choice(
                image_ids[lesion_id],
                min(ncols, image_ids[lesion_id].shape[0]),
                replace=False,
            )
            for lesion_id in rand_lesion_ids
        }
        # Re-define ncols to avoid empty columns (maybe rand_image_ids[lesion_id] < ncols for all lesion_id).
        ncols = max(len(v) for v in rand_image_ids.values())

        # Set up our figure and axes...
        fig, ax = plt.subplots(nrows, ncols)
        fig.set_figheight(8 * nrows)
        fig.set_figwidth(5 * ncols)
        fig.subplots_adjust(hspace=partitions + predictions)

        for i in range(nrows):
            for j in range(ncols):
                try:
                    lesion_id = rand_lesion_ids[i]
                    image_id = rand_image_ids[lesion_id][j]
                    # Caption for the image.
                    # What basic medatdata do we have for the image?
                    caption = view_images.caption_info(image_id, df=df)
                    # What partition (train/val/neither) does the image come from, if applicable?
                    if partitions:
                        PART = view_images.caption_part(image_id, df1=df1, df2=df2)
                        caption += "\n" + PART
                    else:
                        pass
                    # What are our models' predictions about the image, if applicable?
                    if predictions:
                        PRED = view_images.caption_pred(image_id, df1=df1, df2=df2)
                        caption += "\n" + PRED
                    else:
                        pass
                    # Set up the axes with no ticks and caption as label.
                    ax = plt.subplot(nrows, ncols, i * ncols + j + 1)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_xlabel(caption, fontsize=15, loc="left")

                    # Open the image
                    image = Image.open(path.joinpath(f"{image_id}.jpg"))
                    # Show the image
                    plt.imshow(image)

                # If no image to show, leave an empty space in the grid.
                except:
                    ax = plt.subplot(nrows, ncols, i * ncols + j + 1)
                    plt.axis("off")
        plt.show()

    @staticmethod
    def one_dx_per_column(
        dxs: list,
        df: pd.DataFrame,
        path: Path,
        df1: pd.DataFrame = None,
        df2: pd.DataFrame = None,
        nrows: int = 10,
        partitions: bool = False,
        predictions: bool = False,
    ) -> None:
        if predictions:
            info = [
                "Below each image are probabilities given by our binary classification models.",
                "To the left of ¦ is the probability according to a model trained on exactly one image per lesion in our training set.",
                "To the right of ¦ is the probability according to a model trained on all images of each lesion in our training set.",
                "For instance, 'mel: 10% ¦ 15%' means that our mel/not binary classifiers determined that the probability that the lesion is melanoma is 10% (one image/lesion) and 15% (multiple images/lesion).",
                "\nNote that we're viewing original images here, not the augmented images the models were trained on.\n",
            ]
            for line in info:
                print(line)

        rand_lesion_ids = {
            dx: np.random.choice(
                df[df["dx"] == dx]["lesion_id"].unique(), nrows, replace=False
            )
            for dx in dxs
        }
        rand_image_ids = {
            dx: [
                np.random.choice(
                    df[df["lesion_id"] == lesion_id]["image_id"].values, 1
                )[0]
                for lesion_id in rand_lesion_ids[dx]
            ]
            for dx in dxs
        }

        ncols = len(dxs)

        fig, ax = plt.subplots(nrows, ncols)
        fig.set_figheight(8 * nrows)
        fig.set_figwidth(5 * ncols)
        fig.subplots_adjust(hspace=partitions + predictions)

        for i in range(nrows):
            for j, dx in enumerate(dxs):
                try:
                    image_id = rand_image_ids[dx][i]
                    # Caption for the image.
                    # What basic medatdata do we have for the image?
                    caption = view_images.caption_info(image_id, df=df)
                    # What partition (train/val/neither) does the image come from, if applicable?
                    if partitions:
                        PART = view_images.caption_part(image_id, df1=df1, df2=df2)
                        caption += "\n" + PART
                    else:
                        pass
                    # What are our models' predictions about the image, if applicable?
                    if predictions:
                        PRED = view_images.caption_pred(image_id, df1=df1, df2=df2)
                        caption += "\n" + PRED
                    else:
                        pass
                    # Set up the axes with no ticks and caption as label.
                    ax = plt.subplot(nrows, ncols, i * ncols + j + 1)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_xlabel(caption, fontsize=18, loc="left")

                    # Open the image
                    image = Image.open(path.joinpath(f"{image_id}.jpg"))
                    # Show the image
                    plt.imshow(image)

                # If no image to show, leave an empty space in the grid.
                except:
                    ax = plt.subplot(nrows, ncols, i * ncols + j + 1)
                    plt.axis("off")
        plt.show()


class analyse_metadata:
    @staticmethod
    def frequencies(df: pd.DataFrame, *args: Union[str, int]) -> pd.DataFrame:
        """
        Either args = (col1) or args = (col1, value, col2), where col1
        is a column name ('dx', 'dx_type', etc.), value is a possible
        value within the corresponding column (e.g. 'mel', 'histo',
        etc.), and col2 is another column name (e.g. 'age', 'sex',
        'localization').

        Output:
            In case args = (col1), output gives the frequencies (absolute
            and relative) for all of the different values found in the
            relevant dataframe under column col1. In case args = (col1,
            value, col2), we restrict the relevant dataframe to rows where
            only value appears under column col1, and in _that_ restricted
            dataframe, output gives the frequencies (absolute and relative)
            of all of the different values under col2.
        """
        if len(args) == 1:
            col = df[args[0]]
            output = pd.concat(
                [
                    col.value_counts(dropna=False),
                    col.value_counts(normalize=True, dropna=False).mul(100).round(2),
                ],
                axis=1,
                keys=["freq", "%"],
            )
            output.index.names = [args[0]]
            return output.T
        elif len(args) == 3:
            col1, value, col2 = args
            col = df[df[col1] == value][col2]
            output = pd.concat(
                [
                    col.value_counts(dropna=False),
                    col.value_counts(normalize=True, dropna=False).mul(100).round(2),
                ],
                axis=1,
                keys=["freq", "%"],
            )
            output.index.names = [col2]
            return output.T
        else:
            raise ValueError(
                "Invalid number of arguments. Expected 1 (col1) or 3 (col1, value, col2) arguments."
            )
