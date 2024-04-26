import pandas as pd
import numpy as np
from typing import Union, Tuple, Callable
import copy

# from processing import process
# from pathlib import Path
# from multiclass_models import resnet18

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    balanced_accuracy_score,
    fbeta_score,
    matthews_corrcoef,
)

def metric_dictionary(target: np.ndarray, 
                      prediction: np.ndarray,
                      probabilities: Union[np.ndarray, None] = None,) -> dict:
    metrics = ["ACC",
              "BACC",
              "precision",
              "recall",
              "F1/2",
              "F1",
              "F2",
              "MCC",
              "ROC-AUC mac",
              "ROC-AUC wt",
              "ROC-AUC wt*"]

    metric_dict = { metric : [] for metric in metrics}

    try:
        accuracy = accuracy_score(target, prediction)
    except:
        accuracy = np.nan
        
    try:
        balanced_acc = balanced_accuracy_score(target, prediction)
    except:
        balanced_acc = np.nan
    try:
        precision = precision_score(target, prediction, average="macro", zero_division=np.nan)
    except:
        precision = np.nan
    try:
        recall = recall_score(target, prediction, average="macro", zero_division=np.nan)
    except:
        recall = np.nan
    try:
        fhalf = fbeta_score(target, prediction, beta=1/2, average="macro")
    except:
        fhalf = np.nan
    try:
        f1 = f1_score(target, prediction, average="macro")
    except:
        f1 = np.nan
    try:
        f2 = fbeta_score(target, prediction, beta=2, average="macro")
    except:
        f2 = np.nan
    try:
        mcc = matthews_corrcoef(target, prediction)
    except:
        mcc = np.nan
        
    if probabilities is None:
        roc_auc_mac = np.nan
        roc_auc_wt = np.nan
        roc_auc_wt_star = np.nan
    else:
        if probabilities.shape[1] == 2:
            try:
                roc_auc_mac = roc_auc_score(target, probabilities[:, 1], average="macro")
            except:
                roc_auc_mac = np.nan
            try:
                roc_auc_wt = roc_auc_score(target, probabilities[:, 1], average="weighted")
            except:
                roc_auc_wt = np.nan
            try:            
                class_counts = np.bincount(target)
                class_weights = 1 / class_counts
                roc_auc_wt_star = roc_auc_score(target, probabilities[:, 1], average="weighted", sample_weight=class_weights[target])
            except:
                roc_auc_wt_star = np.nan
        elif probabilities.shape[1] > 2:
            try:
                roc_auc_mac = roc_auc_score(target, probabilities, average="macro", multi_class="ovr")
            except:
                roc_auc_mac = np.nan
            try:
                roc_auc_wt = roc_auc_score(target, probabilities, average="weighted", multi_class="ovr")
            except:
                roc_auc_wt = np.nan
            try:            
                class_counts = np.bincount(target)
                class_weights = 1 / class_counts
                roc_auc_wt_star = roc_auc_score(target, probabilities, average="weighted", sample_weight=class_weights[target], multi_class="ovr")
            except:
                roc_auc_wt_star = np.nan            
              
    metric_dict["ACC"].append(accuracy)
    metric_dict["BACC"].append(balanced_acc)
    metric_dict["precision"].append(precision)
    metric_dict["recall"].append(recall)
    metric_dict["F1/2"].append(fhalf)
    metric_dict["F1"].append(f1)    
    metric_dict["F2"].append(f2)    
    metric_dict["MCC"].append(mcc)
    metric_dict["ROC-AUC mac"].append(roc_auc_mac)
    metric_dict["ROC-AUC wt"].append(roc_auc_wt)
    metric_dict["ROC-AUC wt*"].append(roc_auc_wt_star)

    return metric_dict







def weighted_average_f(beta: Union[float, None], 
                       weights: Union[np.ndarray, None], 
                       precision: Union[np.ndarray, pd.DataFrame], 
                       recall: Union[np.ndarray,pd.DataFrame]) -> float:
    
    assert np.ravel(precision).shape == np.ravel(recall).shape, 'precision and recall arrays must be of the same length'
    
    if precision.all() == 0 or recall.all() == 0:
        return np.nan
    
    if beta is None:
        beta = 1
    
    if weights is None:
        weights = np.ones(np.ravel(precision).shape)
        
    assert np.ravel(precision).shape == np.ravel(weights).shape, 'weights must have the same length as precision and recall'
    if weights.all() == 0:
        return np.nan
    
    fbeta = (1 + beta**2)/(1/precision + beta**2/recall)
    
    weighted_average_fbeta = np.sum(weights*fbeta)/np.sum(weights)
    
    return weighted_average_fbeta

def custom_sort_key(x):
    if isinstance(x, (int, float)):
        return (0, x)  # Sort integers and floats together
    elif isinstance(x, str):
        return (1, x)  # Sort strings
    elif x is None:
        return (2, None)  # Sort None values last
    else:
        return (3, str(x))  # Sort other types last, convert to string for sorting

def sort_mixed_type_list(lst):
    return sorted(lst, key=custom_sort_key)

def pad_df(df: pd.DataFrame, lst: Union[list, None], full: Union[bool,None] = None) -> pd.DataFrame:  
       
    if lst is None:
        lst = {}
    else:
        lst = set(lst)
     
    if full is None:
        full = True
        
    row_col_union = set(df.index).union(set(df.columns)).union(lst)  
        
    if full:
        row_col_union_integers = {x for x in row_col_union if isinstance(x, int)} 
        row_col_union_non_integers = {x for x in row_col_union if not isinstance(x,int)} 
        m = min(row_col_union_integers)
        M = max(row_col_union_integers)
        all_rows_and_cols = list(range(m,M + 1)) + list(row_col_union_non_integers)
    else:
        all_rows_and_cols = list(row_col_union)

    missing_columns = [col for col in all_rows_and_cols if col not in df.columns]
    missing_df = pd.DataFrame(0, index=df.index, columns=missing_columns)
    output_df = pd.concat([df, missing_df], axis=1)

    missing_rows = [row for row in all_rows_and_cols if row not in df.index]
    missing_df = pd.DataFrame(0, index=missing_rows, columns=output_df.columns)
    output_df = pd.concat([output_df, missing_df])

    sorted_columns = sort_mixed_type_list(output_df.columns)
    sorted_rows = sort_mixed_type_list(output_df.index)

    output_df = output_df[sorted_columns]
    output_df = output_df.loc[sorted_rows]
    
    return output_df

def confusion_matrix_with_metric(AxB: Union[pd.DataFrame,dict],
                                 lst: Union[list,None] = None,
                                 full_pad: Union[bool,None] = None,
                                 func: Union[None, float,
                                             Callable[[pd.DataFrame,
                                                 Union[float, None],
                                                 Union[np.ndarray, None]], float]] = None,
                                 beta: Union[float, None] = None,
                                 weights: Union[np.ndarray, None] = None,
                                 percentage: bool = False,
                                 map_labels: Union[dict, None] = None,) -> pd.DataFrame:        
    
    if full_pad is None:
        full_pad = True
    
    if lst is None and map_labels is not None:
        lst = list(map_labels.keys())
    
    if type(AxB) == dict:
        AxB_df = pd.DataFrame(AxB)
        df = pad_df(AxB_df, lst, full_pad)
    else:
        df = pad_df(AxB, lst, full_pad)
    
    # AxB is supposed to be the result of pd.crosstab(A, B, margins=True).
    # Let's check to see whether both margins are included.
    # If not, add them.
    # Could be some corner cases when the last column of AxB sans margin just so happens to be the sum of the previous columns.
    # We won't worry about that at this stage.

    is_last_column_sum = df.iloc[:, :-1].sum(axis=1).equals(df.iloc[:, -1])
    if not is_last_column_sum:
        df['All'] = df.sum(axis=1)

    is_last_row_sum = df.iloc[:-1, :].sum(axis=0).equals(df.iloc[-1, :])
    if not is_last_row_sum:
        df.loc['All'] = df.sum(axis=0)    

    diagonal_entries = np.diag(df.values)
    # Add a recall column 
    last_col_name = df.columns[-1]
    df['recall'] = diagonal_entries/df[last_col_name]

    # And a precision row
    diagonal_entries = np.append(diagonal_entries, [0])
    df.loc['precision'] = diagonal_entries/df.iloc[-1]

    # Extract the recall and precision arrays
    recall = df['recall'][:-2]
    precision = df.loc['precision'][:-2] 

    # Calculate the macro function (whatever it is) of precision and recall and put it in the bottom right corner of df
    if func is None:
        df.iloc[-1, -1] = np.nan
    elif isinstance(func, (float,int)):
        df.iloc[-1, -1] = func
    else:
        try:
            df.iloc[-1, -1] = func(beta, weights, precision, recall)
        except Exception as e:
            print(f"Error calculating metric: {e}")

    # df should have originally been all integers, but now we have floats, so let's format the original part
    df.iloc[:-1,:-1,] = df.iloc[:-1,:-1,].applymap(lambda x : f"{int(x):,}")

    # Insert blanks where we have no meaningful values
    df.iloc[-1, -2] = np.nan
    df.iloc[-2, -1] = np.nan

    # Label the index and the columns
    df = df.rename_axis(index='actual', columns='predicted')
    
    if map_labels is not None:
        column_mapping = { k : v for k, v in map_labels.items() if k in df.columns }
        column_mapping.update({ k : k for k in df.columns if k not in map_labels.keys()})
        row_mapping = { k : v for k, v in map_labels.items() if k in df.index }
        row_mapping.update({ k : k for k in df.index if k not in map_labels.keys()})
        
        df.columns = df.columns.map(column_mapping)
        df.index = df.index.map(row_mapping)   
    
    return df

def display_confusion_matrices(which_model: dict,
                               func: Callable[[pd.DataFrame,
                                               Union[float, None],
                                               Union[np.ndarray, None]], float] = None,
                              beta: Union[float, None] = None,
                              weights: Union[np.ndarray, None] = None) -> None:

    if which_model["confusion"] is None:
        return
   
    if func is None:
        for i, d in enumerate(which_model["confusion"]):
            if type(d) == dict:
                d_df = pd.DataFrame(d)
                display(i, d_df)
        
    else:
        for i, d in enumerate(which_model["confusion"]):
            if beta is None:
                beta = 2
            if weights is None:
                weights = np.ones(pd.DataFrame(d).shape)
                
            d_augmented = func(d, beta = beta, weights = weights)
            display(i, d_augmented.fillna('_'))
   

def custom_confusion(
    labels: pd.Series,
    predictions: pd.Series,
    label_dictionary: dict = None,
    dropna: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not dropna:
        # Replace NaN values with a placeholder before computing crosstabs
        labels = labels.fillna("NaN")
        predictions = predictions.fillna("NaN")

    # Standard crosstabs with margins
    # Convention: ground truth labels along rows, predicted labels along columns
    labels_by_predictions = pd.crosstab(
        labels, predictions, dropna=dropna, margins=True
    )
    # Normalizing by 'columns' will give probability of label given a prediction, e.g. probability of actually having melanoma given the prediction is nevus.
    prob_label_given_prediction = (
        pd.crosstab(
            labels, predictions, normalize="columns", dropna=dropna, margins=True
        )
        .mul(100)
        .round(2)
    )
    # Normalizing by 'rows' (or 'index') will give probability of prediction given a label, e.g. probability of predicting melanoma given oen actually has a nevus.
    prob_prediction_given_label = (
        pd.crosstab(labels, predictions, normalize="index", dropna=dropna, margins=True)
        .mul(100)
        .round(2)
    )

    # The normalized crosstab will not contain column sums.
    # Even each column sums to 100%, we still want this:
    column_sums = pd.DataFrame(
        prob_label_given_prediction.sum(axis=0).round().astype(int), columns=["All"]
    ).T
    prob_label_given_prediction_with_sum = pd.concat(
        [prob_label_given_prediction, column_sums]
    )

    row_sums = pd.DataFrame(
        prob_prediction_given_label.sum(axis=1).round().astype(int), columns=["All"]
    ) 
    prob_prediction_given_label_with_sum = pd.concat([prob_prediction_given_label, row_sums], axis = 1)

    # Now we merge the two label_given_prediction crosstabs, displaying (a,b) in each cell, where a is an integer and b is a percentage.
    label_given_prediction_combined_df = pd.merge(
        labels_by_predictions,
        prob_label_given_prediction_with_sum,
        left_index=True,
        right_index=True,
    )
    for column in label_given_prediction_combined_df.columns:
        if column.endswith("_x"):
            column_name = column[:-2]
            label_given_prediction_combined_df[column_name] = list(
                zip(
                    label_given_prediction_combined_df[column_name + "_x"],
                    label_given_prediction_combined_df[column_name + "_y"],
                )
            )

    filtered_label_given_prediction_combined_df = (
        label_given_prediction_combined_df.filter(regex="^(?!.*(_x|_y)$)")
    )

    # Convert string column names to integers
    converted_columns = pd.to_numeric(filtered_label_given_prediction_combined_df.columns, errors='coerce')
    # Replace NaN values with the original column names
    filtered_label_given_prediction_combined_df.columns = converted_columns.where(pd.notnull(converted_columns), filtered_label_given_prediction_combined_df.columns)
    # Convert float column names to integers
    filtered_label_given_prediction_combined_df.columns = [int(col) if isinstance(col, float) else col for col in filtered_label_given_prediction_combined_df.columns]

    # And similar for the prediction_given_label crosstabs...
    prediction_given_label_combined_df = pd.merge(
        labels_by_predictions,
        prob_prediction_given_label_with_sum,
        left_index=True,
        right_index=True,
    )
    for column in prediction_given_label_combined_df.columns:
        if column.endswith("_x"):
            column_name = column[:-2]
            prediction_given_label_combined_df[column_name] = list(
                zip(
                    prediction_given_label_combined_df[column_name + "_x"],
                    prediction_given_label_combined_df[column_name + "_y"],
                )
            )

    filtered_prediction_given_label_combined_df = (
        prediction_given_label_combined_df.filter(regex="^(?!.*(_x|_y)$)")
    )
    # Convert string column names to integers
    converted_columns_2 = pd.to_numeric(filtered_prediction_given_label_combined_df.columns, errors='coerce')
    # Replace NaN values with the original column names
    filtered_prediction_given_label_combined_df.columns = converted_columns_2.where(pd.notnull(converted_columns_2), filtered_prediction_given_label_combined_df.columns)
    # Convert float column names to integers
    filtered_prediction_given_label_combined_df.columns = [int(col) if isinstance(col, float) else col for col in filtered_prediction_given_label_combined_df.columns]

    # Finally, we format so that (a,b) is displayed as a (↓b%) for the label given prediction probabilities.
    # (The down arrow aids the viewer, indicating that one should be looking down a given column (prediction)).
    def format_label_given_prediction(a, b):
        return f"{a} (↓{b}%)"

    label_given_prediction_formatted_df = (
        filtered_label_given_prediction_combined_df.applymap(
            lambda x: format_label_given_prediction(*x) if isinstance(x, tuple) else x
        )
    )
    label_given_prediction_formatted_df.columns.name = "pred"
    label_given_prediction_formatted_df.index.name = "label"
    # For prediction given label, we look along a row, hence the right arrow.
    def format_prediction_given_label(a, b):
        return f"{a} (→{b}%)"

    prediction_given_label_formatted_df = (
        filtered_prediction_given_label_combined_df.applymap(
            lambda x: format_prediction_given_label(*x) if isinstance(x, tuple) else x
        )
    )
    
    prediction_given_label_formatted_df.columns.name = "pred"
    
    # Depending on the situation, we might want numbers, percentages, or the combined table:
    output = {}
    output["labels_by_predictions"] = labels_by_predictions
    output["prob_label_given_prediction"] = prob_label_given_prediction
    output["prob_prediction_given_label"] = prob_prediction_given_label
    output["label_given_prediction_merged"] = label_given_prediction_formatted_df
    output["prediction_given_label_merged"] = prediction_given_label_formatted_df

    if label_dictionary is not None:
        for key, value in output.items():
            value.rename(index=label_dictionary, inplace=True)
            value.rename(columns=label_dictionary, inplace=True)

    return output