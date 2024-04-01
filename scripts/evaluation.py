import pandas as pd
import numpy as np
from typing import Tuple

   

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