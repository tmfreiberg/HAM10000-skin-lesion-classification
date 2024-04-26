import os
from pathlib import Path
from typing import Union
import inspect
import pandas as pd
import matplotlib.pyplot as plt

# SETTING PATHS IN A PLATFORM-INDEPENDENT WAY

gbl_project_path_name = "SKIN_LESION_CLASSIFICATION"
gbl_project_path = Path(os.getenv(gbl_project_path_name))

class path_setup:

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
        if os.getenv(project_path_name) is None:
            print(f"\nThe environment variable '{project_path_name}' does not exist.")
            return path_setup.how(project_path_name, os.name)
        elif not os.path.isdir(os.getenv(project_path_name)):
            print(
                f"\nThe environment variable '{project_path_name}' does not point to a valid directory path."
            )
            return path_setup.how(project_path_name, os.name)
        else:
            print(
                f"\nThe environment variable '{project_path_name}' contains a valid directory path.\n"
            )
            global gbl_project_path_name 
            gbl_project_path_name = project_path_name
            global gbl_project_path
            gbl_project_path = Path(os.getenv(gbl_project_path_name))
            print("=" * 3 + f"\nproject_path = {gbl_project_path}\n" + "=" * 3)
            return gbl_project_path

    @staticmethod
    def change(project_path_name: str):
        new_project_path_name = input("\nEnter new path name, or 'c' to cancel: ")
        if new_project_path_name.lower() == "c":
            return path_setup.check(project_path_name)
        else:
            project_path_name = new_project_path_name
            return path_setup.confirm(project_path_name)

    @staticmethod
    def confirm(project_path_name: str = gbl_project_path_name):
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
        base_path: Path = gbl_project_path, base_name: str = "project", Print: bool = True
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

# USEFUL FUNCTIONS

def display(*args):
    try:
        # Check if running in a Jupyter Notebook environment
        from IPython.display import display as ipy_display

        # Display each argument using IPython display function
        for arg in args:
            ipy_display(arg)
    except ImportError:
        # If IPython is not available, just print each argument
        for arg in args:
            print(arg)

def clear_output():
    # Check if running in a Jupyter Notebook environment
    try:
        from IPython.display import clear_output as ipy_clear_output

        if ipy_clear_output is not None:
            ipy_clear_output(wait=True)
            return
    except ImportError:
        pass  # Continue if IPython is not available or running in a script

    # Check the operating system and clear output accordingly
    if os.name == "nt":  # Windows
        os.system("cls")
    else:  # Unix-based systems (Linux, macOS)
        os.system("clear")

    return

def get_variable_name(var):
    # Get the name of the variable using the inspect module
    for name, value in inspect.currentframe().f_back.f_locals.items():
        if value is var:
            return name
    return None  # Variable name not found

def save_df(df: pd.DataFrame, save_as: Union[str, Path], tack_on: Union[str, None] = None) -> None:
    # If filename is a Path object, convert it to a string
    if isinstance(save_as, str):
        save_as = Path(save_as)
        
    # Insert suffix before the file extension
    if tack_on is not None:
        save_as = save_as.with_name(f"{save_as.stem}_{tack_on}{save_as.suffix}")

    # Determine the file extension
    file_extension = save_as.suffix.lower()

    # Choose the saving method based on the file extension
    if file_extension == '.csv':
        df.to_csv(save_as, index=False)
    elif file_extension == '.xlsx':
        df.to_excel(save_as, index=False)
    elif file_extension == '.html':
        df.to_html(save_as, index=False)
    elif file_extension == '.json':
        df.to_json(save_as, orient='records')
    elif file_extension == '.png':
        plt.axis('off')  # Turn off the axis
        plt.table(cellText=df.values, colLabels=df.columns, loc='center')  # Display DataFrame as a table
        plt.savefig(save_as)  # Save as PNG
    elif file_extension == '.txt':
        # Convert DataFrame to dictionary
        df_dict = df.to_dict()
        # Save dictionary as text file
        with open(save_as, 'w') as file:
            file.write(str(df_dict))
    else:
        print(f"Unsupported file extension: {file_extension}. Please use csv, xlsx, html, or json.")
        
def print_header(header: str) -> None:
    length = len(header)
    output_sequence = ["", "="*length, header.upper(), "="*length, ""]
    output = '\n'.join(output_sequence)
    print(output) 