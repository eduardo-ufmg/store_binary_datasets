import logging
import os
from typing import cast

import numpy as np
import pandas as pd
from sklearn.datasets import (fetch_openml, load_breast_cancer, load_digits,
                              load_iris)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class SklearnLoadedDataset:
    """
    Container for datasets loaded from scikit-learn.
    This class encapsulates the data, target labels, feature names, and target names
    from a dataset loaded using scikit-learn's dataset loaders. It provides a convenient
    structure for storing and accessing these components together.

    Attributes:
        data (np.ndarray): The feature matrix of shape (n_samples, n_features).
        target (np.ndarray): The target labels of shape (n_samples,).
        feature_names (list): List of feature names corresponding to columns in `data`.
        target_names (list): List of target class names.

    Methods:
        __init__(data: np.ndarray, target: np.ndarray, feature_names: list, target_names: list):
            Initialize the dataset container with data, target, feature names, and target names.
    """

    data: np.ndarray
    target: np.ndarray
    feature_names: list
    target_names: list

    def __init__(
        self,
        data: np.ndarray,
        target: np.ndarray,
        feature_names: list,
        target_names: list,
    ):
        self.data = data
        self.target = target
        self.feature_names = feature_names
        self.target_names = target_names


def save_dataset(X: np.ndarray, y: np.ndarray, name: str, base_dir: str):
    """
    Saves the dataset (X, y) to the specified directory.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Label vector.
        name (str): Name of the dataset (e.g., 'breast_cancer').
        base_dir (str): The base directory to save the 'sets' folder.
    """
    sets_dir = os.path.join(base_dir, "sets")
    os.makedirs(sets_dir, exist_ok=True)

    filepath = os.path.join(sets_dir, f"{name}.npz")
    try:
        np.savez_compressed(filepath, X=X, y=y)
        logger.info(f"Dataset '{name}' saved successfully to '{filepath}'")
        logger.info(
            f"X shape: {X.shape}, y shape: {y.shape}, y unique values: {np.unique(np.asarray(y), return_counts=True)}"
        )
    except Exception as e:
        logger.error(f"Error saving dataset '{name}': {e}")


def preprocess_data(
    X_df_orig: pd.DataFrame,
    y_series_orig: pd.DataFrame,
    is_target_categorical: bool = True,
    binarize_target_config: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Preprocesses features (handles missing values, scaling, one-hot encoding)
    and ensures target labels are binary (0, 1).

    Args:
        X_df_orig (pd.DataFrame): Original feature DataFrame.
        y_series_orig (pd.Series): Original target Series.
        is_target_categorical (bool): Indicates if the target needs LabelEncoding.
                                      Set to False if y is already 0/1 numeric.
        binarize_target_config (dict, optional): Configuration for binarizing a multi-class target.
            E.g., {'type': 'one_vs_rest', 'positive_class': 'some_label'}
                  {'type': 'select_two', 'class1': 'labelA', 'class2': 'labelB', 'map_to_0': 'labelA'}

    Returns:
        tuple: (X_processed_np, y_processed_np)
               Processed feature matrix as a NumPy array and
               processed label vector as a NumPy array.
    """
    X_df = X_df_orig.copy()
    y_series = y_series_orig.copy()

    # --- Apply binarize_target_config early if it involves filtering X and y ---
    if binarize_target_config:
        if binarize_target_config["type"] == "one_vs_rest":
            pos_class = binarize_target_config["positive_class"]
            y_series = y_series.apply(lambda x: 1 if x == pos_class else 0)
        elif binarize_target_config["type"] == "select_two":
            c1 = binarize_target_config["class1"]
            c2 = binarize_target_config["class2"]
            map_to_0 = binarize_target_config.get("map_to_0", c1)

            # Filter for only the two classes
            mask = y_series.isin([c1, c2])

            y_series_before_filter_len = len(y_series)
            X_df = X_df[mask]
            y_series = y_series[mask]

            if X_df.shape[0] == 0:  # All samples dropped
                raise ValueError(
                    f"No samples remaining after filtering for classes {c1}, {c2} in 'select_two' binarization. Original y had {y_series_before_filter_len} samples."
                )

            y_series = y_series.apply(lambda x: 0 if x == map_to_0 else 1)
        is_target_categorical = False  # Target is now 0/1

    # --- Preprocess Features (X) from potentially filtered X_df ---
    numerical_cols = X_df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X_df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    X_processed_parts = []

    if numerical_cols:
        num_imputer = SimpleImputer(strategy="median")
        X_numerical = X_df[numerical_cols]
        X_numerical_imputed = num_imputer.fit_transform(X_numerical)
        X_processed_parts.append(X_numerical_imputed)
        logger.debug(f"Processed numerical features: {numerical_cols}")

    if categorical_cols:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        X_categorical = X_df[categorical_cols]
        X_categorical_imputed = cat_imputer.fit_transform(X_categorical)
        X_categorical_imputed_df = pd.DataFrame(
            X_categorical_imputed, columns=categorical_cols, index=X_df.index
        )
        X_categorical_encoded = pd.get_dummies(
            X_categorical_imputed_df, columns=categorical_cols, dummy_na=False
        )
        X_processed_parts.append(X_categorical_encoded.values)
        logger.debug(
            f"Processed categorical features (one-hot encoded): {categorical_cols}"
        )

    if not X_processed_parts:
        if X_df_orig.shape[1] > 0:  # Original dataframe had columns
            logger.error(
                "No features were identified for processing (neither numerical nor categorical) from the input X_df."
            )
        else:  # Original dataframe was empty
            logger.error("Input X_df was empty. No features to process.")
        raise ValueError("No features available for processing.")

    try:
        X_processed_np = np.concatenate(X_processed_parts, axis=1).astype(np.float64)
    except ValueError as e:
        logger.error(
            f"Error during feature concatenation: {e}. This might mean all feature parts were empty or incompatible."
        )
        raise ValueError(f"Could not concatenate processed feature parts: {e}")

    if X_processed_np.shape[0] == 0 and X_df_orig.shape[0] > 0:
        # This can happen if X_df became empty due to y-filtering, and y also became empty.
        raise ValueError(
            "All samples were removed during preprocessing (e.g. target binarization), resulting in empty X and y."
        )

    # --- Check for features that failed numerical encoding (non-finite values) ---
    if X_processed_np.shape[1] > 0:  # Only if there are columns to check
        finite_cols_mask = np.all(np.isfinite(X_processed_np), axis=0)
        if not np.all(finite_cols_mask):
            num_total_features = X_processed_np.shape[1]
            X_processed_np = X_processed_np[:, finite_cols_mask]
            num_dropped_features = num_total_features - X_processed_np.shape[1]

            logger.warning(
                f"Dropped {num_dropped_features} feature(s) out of {num_total_features} "
                f"due to non-finite values (NaN or infinity) after processing."
            )
            if X_processed_np.shape[1] == 0:
                logger.error(
                    "All features were dropped due to non-finite values. Cannot proceed with this dataset."
                )
                raise ValueError(
                    "All features resulted in non-finite values and were dropped."
                )
    elif (
        X_df_orig.shape[1] > 0
    ):  # Original X had features, but now X_processed_np has no columns
        logger.error(
            "No features remaining after processing (e.g., all columns were of unsupported types or dropped)."
        )
        raise ValueError("Resulting feature matrix X has no columns after processing.")

    # --- Preprocess Target (y) ---
    y_np = np.asarray(
        y_series.values
    )  # y_series might have been filtered or transformed

    if is_target_categorical:
        if (
            y_np.shape[0] == 0
        ):  # y_series became empty (e.g. due to binarize_target_config that removed all samples)
            raise ValueError(
                "Target variable y is empty before LabelEncoding. Cannot proceed."
            )
        label_encoder = LabelEncoder()
        y_processed_interim = label_encoder.fit_transform(y_np)
        unique_labels_encoded = np.unique(y_processed_interim)

        if len(unique_labels_encoded) > 2:
            logger.warning(
                f"Target has {len(unique_labels_encoded)} classes after LabelEncoding: {unique_labels_encoded} (from originals: {np.unique(y_np)}). Attempting to use first two encoded classes."
            )

            map_to_0_val = unique_labels_encoded[0]
            map_to_1_val = unique_labels_encoded[1]

            y_processed_np_final = np.full_like(y_processed_interim, -1, dtype=int)
            y_processed_np_final[y_processed_interim == map_to_0_val] = 0
            y_processed_np_final[y_processed_interim == map_to_1_val] = 1

            valid_indices = y_processed_np_final != -1
            if not np.any(valid_indices):
                raise ValueError(
                    f"All samples dropped when binarizing multi-class target from {len(unique_labels_encoded)} classes. No samples matched the first two selected encoded classes."
                )

            y_processed_np = y_processed_np_final[valid_indices]
            X_processed_np = X_processed_np[valid_indices]  # Filter X accordingly

            if X_processed_np.shape[0] == 0 or y_processed_np.shape[0] == 0:
                raise ValueError(
                    f"All samples dropped after attempting to binarize target from {len(unique_labels_encoded)} classes. Original unique encoded: {unique_labels_encoded}"
                )
        elif (
            len(unique_labels_encoded) == 2
        ):  # Already two classes, ensure they become 0 and 1
            # This will be handled by the final check to map to 0,1 if not already.
            y_processed_np = y_processed_interim
        elif len(unique_labels_encoded) == 1:
            raise ValueError(
                f"Target has only one class after LabelEncoding: {unique_labels_encoded} (from original: {np.unique(y_np)}). Not a binary problem."
            )
        else:  # 0 classes
            raise ValueError(
                f"Target has no classes after LabelEncoding (original y might have been empty or all NaNs not caught). Original unique: {np.unique(y_np)}"
            )
    else:  # is_target_categorical is False (y is supposedly numeric, possibly 0/1)
        if isinstance(y_np, (pd.Series, pd.DataFrame)):  # Ensure it's a numpy array
            y_np = y_np.values
        try:
            y_processed_np = y_np.astype(int)
        except ValueError as e:
            raise ValueError(
                f"Target variable could not be cast to int (is_target_categorical=False). Values: {np.unique(np.array(y_np))}. Error: {e}"
            )

    # --- Final check and enforcement for binary target (y must be 0 and 1) ---
    y_processed_np = np.asarray(y_processed_np)
    if len(y_processed_np) == 0:
        # This can happen if all samples were filtered out during target processing
        # and X_processed_np might also be empty (checked earlier for X).
        raise ValueError(
            "Target variable y is empty after all processing. Cannot create a dataset."
        )

    unique_y_values = np.unique(y_processed_np)

    if len(unique_y_values) == 2:
        # If classes are already [0, 1], great. Otherwise, map them.
        if not (0 in unique_y_values and 1 in unique_y_values):
            logger.warning(
                f"Target labels are {unique_y_values}. Remapping to 0 and 1 (mapping {unique_y_values[0]} to 0, {unique_y_values[1]} to 1)."
            )
            # Map the numerically smaller unique value to 0, and the larger to 1
            val_a, val_b = unique_y_values[0], unique_y_values[1]
            y_processed_np = np.where(y_processed_np == val_a, 0, 1)
            # Verify remapping
            remapped_unique_y = np.unique(y_processed_np)
            if not (
                len(remapped_unique_y) == 2
                and 0 in remapped_unique_y
                and 1 in remapped_unique_y
            ):
                logger.error(
                    f"Target remapping failed. Labels are now: {remapped_unique_y}. Expected [0, 1]."
                )
                raise ValueError(
                    f"Target remapping from two classes ({unique_y_values}) to [0,1] failed."
                )
    elif len(unique_y_values) == 1:
        logger.warning(
            f"Target variable has only one unique class after all processing: {unique_y_values}. Dataset will be skipped as it's not binary."
        )
        raise ValueError(
            f"Target has only one class ({unique_y_values[0]}), not suitable for binary classification."
        )
    else:  # 0 classes or more than 2 classes
        logger.warning(
            f"Target variable does not have exactly two unique classes after processing. Found {len(unique_y_values)} classes: {unique_y_values}. Dataset will be skipped."
        )
        raise ValueError(
            f"Target is not binary. Found {len(unique_y_values)} unique classes: {unique_y_values}."
        )

    # Final check on X and y shapes (must have same number of samples)
    if X_processed_np.shape[0] != len(y_processed_np):
        raise ValueError(
            f"Mismatch in number of samples between X ({X_processed_np.shape[0]}) and y ({len(y_processed_np)}) after processing."
        )

    return X_processed_np, y_processed_np


def main():
    """
    Loads, processes, and saves multiple datasets.
    """
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()
        logger.warning(
            f"'__file__' not defined. Using current working directory: {script_dir}"
        )

    base_storage_dir = os.path.join(script_dir, ".")
    logger.info(
        f"Base storage directory for 'sets': {os.path.abspath(base_storage_dir)}"
    )

    datasets_to_process = []

    # --- 1. Scikit-learn Datasets ---
    logger.info("--- Processing Scikit-learn Datasets ---")
    cancer = cast(SklearnLoadedDataset, load_breast_cancer())
    X_cancer_df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    y_cancer_series = pd.Series(cancer.target)
    datasets_to_process.append(
        {
            "df": X_cancer_df,
            "series": y_cancer_series,
            "name": "breast_cancer",
            "is_target_categorical": False,
        }
    )

    iris = cast(SklearnLoadedDataset, load_iris())
    X_iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    y_iris_series = pd.Series(
        iris.target_names[iris.target]
    )  # Use names for clarity in binarization
    # Iris: class 'setosa' vs. rest
    datasets_to_process.append(
        {
            "df": X_iris_df.copy(),
            "series": y_iris_series.copy(),
            "name": "iris_binary_setosa_vs_rest",
            "binarize_target_config": {
                "type": "one_vs_rest",
                "positive_class": "setosa",
            },
            # is_target_categorical will be effectively False due to binarize_target_config
        }
    )
    # Iris: class 'setosa' vs. 'versicolor'
    datasets_to_process.append(
        {
            "df": X_iris_df.copy(),
            "series": y_iris_series.copy(),
            "name": "iris_binary_setosa_vs_versicolor",
            "binarize_target_config": {
                "type": "select_two",
                "class1": "setosa",
                "class2": "versicolor",
                "map_to_0": "setosa",
            },
        }
    )

    digits = cast(SklearnLoadedDataset, load_digits())
    X_digits_df = pd.DataFrame(digits.data)
    y_digits_series = pd.Series(digits.target)  # Targets are 0-9
    # Digits: 0 vs 1
    datasets_to_process.append(
        {
            "df": X_digits_df.copy(),
            "series": y_digits_series.copy(),
            "name": "digits_binary_0_vs_1",
            "binarize_target_config": {
                "type": "select_two",
                "class1": 0,
                "class2": 1,
                "map_to_0": 0,
            },
        }
    )
    # Digits: 5 vs rest
    datasets_to_process.append(
        {
            "df": X_digits_df.copy(),
            "series": y_digits_series.copy(),
            "name": "digits_binary_5_vs_rest",
            "binarize_target_config": {"type": "one_vs_rest", "positive_class": 5},
        }
    )

    # --- 2. OpenML Datasets ---
    logger.info("--- Processing OpenML Datasets ---")
    openml_datasets_info = [
        {
            "name": "diabetes",
            "id": 37,
            "target_col": "class",
            "is_target_categorical": True,
        },  # tested_positive, tested_negative
        {
            "name": "ionosphere",
            "id": 59,
            "target_col": "class",
            "is_target_categorical": True,
        },  # 'g', 'b'
        {
            "name": "sonar",
            "id": 40,
            "target_col": "Class",
            "is_target_categorical": True,
        },  # 'Rock', 'Mine'
        {
            "name": "spambase",
            "id": 44,
            "target_col": "class",
            "is_target_categorical": False,
        },  # Already 0/1
        {
            "name": "german_credit_g",
            "id": 31,
            "target_col": "class",
            "is_target_categorical": True,
        },  # 'good', 'bad'
        {
            "name": "vote",
            "id": 56,
            "target_col": "Class",
            "is_target_categorical": True,
        },  # 'democrat', 'republican'
        {
            "name": "mushroom",
            "id": 24,
            "target_col": "class",
            "is_target_categorical": True,
        },  # 'e', 'p'
        {
            "name": "banknote-authentication",
            "id": 1462,
            "target_col": "Class",
            "is_target_categorical": False,
        },  # Already 0/1
        {
            "name": "adult",
            "id": 1590,
            "target_col": "class",
            "is_target_categorical": True,
        },  # '>50K', '<=50K'
        {
            "name": "titanic",
            "id": 40945,
            "target_col": "survived",
            "is_target_categorical": True,
        },  # target '0', '1' as strings
        {
            "name": "wpbc",
            "id": 13,
            "target_col": "binaryClass",
            "is_target_categorical": True,
        },  # 'N', 'R'
        {
            "name": "kc1",
            "id": 1067,
            "target_col": "defects",
            "is_target_categorical": True,
        },  # boolean true/false needs handling
        {
            "name": "blood-transfusion-service-center",
            "id": 1464,
            "target_col": "Class",
            "is_target_categorical": True,
        },  # 'donated', 'not donated' (mapped from original int by openml)
        {
            "name": "qsar-biodeg",
            "id": 1494,
            "target_col": "Class",
            "is_target_categorical": True,
        },  # 'ready biodegradable', 'not ready biodegradable'
        {
            "name": "sylvine",
            "id": 1501,
            "target_col": "class",
            "is_target_categorical": False,
        },  # Already 0/1 like
    ]

    for ds_info in openml_datasets_info:
        logger.info(
            f"Fetching and processing OpenML dataset: {ds_info['name']} (ID: {ds_info['id']})"
        )
        try:
            dataset = fetch_openml(data_id=ds_info["id"], as_frame=True, parser="auto")
            X_df = dataset.data
            y_series = dataset.target

            if y_series is None and ds_info["target_col"] in X_df.columns:
                y_series = X_df.pop(ds_info["target_col"])
            elif y_series is None:
                logger.error(
                    f"Target column for {ds_info['name']} could not be identified."
                )
                continue

            # Ensure y_series is a pandas Series
            if not isinstance(y_series, pd.Series):
                y_series = pd.Series(y_series)

            current_is_target_categorical = ds_info["is_target_categorical"]
            current_binarize_config = ds_info.get(
                "binarize_target_config"
            )  # For potential future use

            # Specific handling for 'kc1' boolean target
            if ds_info["name"] == "kc1" and y_series.dtype == "bool":
                y_series = y_series.astype(
                    str
                )  # Convert boolean to string ('True', 'False') for LabelEncoder
                current_is_target_categorical = True  # Ensure LabelEncoder is used

            # Specific handling for 'titanic' target if it's string '0', '1'
            if ds_info["name"] == "titanic":  # target '0', '1' as strings
                unique_titanic_targets = y_series.unique()
                if set(unique_titanic_targets) == {"0", "1"}:
                    y_series = y_series.astype(int)
                    current_is_target_categorical = False
                # else: remains True for LabelEncoder if other values exist

            # Specific handling for 'sylvine' target if it is not binary
            if ds_info["name"] == "sylvine":
                unique_sylvine_targets = y_series.unique()
                if len(unique_sylvine_targets) > 2:
                    logger.warning(
                        f"Target for 'sylvine' has {len(unique_sylvine_targets)} classes: {unique_sylvine_targets}. Will attempt to binarize to first two classes."
                    )
                    current_binarize_config = {
                        "type": "select_two",
                        "class1": unique_sylvine_targets[0],
                        "class2": unique_sylvine_targets[1],
                        "map_to_0": unique_sylvine_targets[0],
                    }
                    current_is_target_categorical = (
                        True  # Will use LabelEncoder for binarization
                    )

            # For OpenML 'blood-transfusion-service-center', target is 'Class' ('donated'/'not donated')
            # but fetch_openml might return it as string '1'/'2'. LabelEncoder will handle this.
            # For 'diabetes', target 'class' is 'tested_positive'/'tested_negative'. LabelEncoder handles.

            datasets_to_process.append(
                {
                    "df": X_df,
                    "series": y_series,
                    "name": ds_info["name"],
                    "is_target_categorical": current_is_target_categorical,
                    "binarize_target_config": current_binarize_config,
                }
            )
        except Exception as e:
            logger.error(
                f"Failed to fetch or initially process OpenML dataset {ds_info['name']} (ID: {ds_info['id']}): {e}"
            )
        logger.info("-" * 30)

    # Process all collected datasets
    for data_dict in datasets_to_process:
        logger.info(f"Processing dataset: {data_dict['name']}")
        try:
            # Convert all column names to string
            data_dict["df"].columns = data_dict["df"].columns.astype(str)

            X_processed, y_processed = preprocess_data(
                data_dict["df"],
                data_dict["series"],
                is_target_categorical=data_dict.get(
                    "is_target_categorical", True
                ),  # Default to True if not specified
                binarize_target_config=data_dict.get("binarize_target_config"),
            )
            save_dataset(X_processed, y_processed, data_dict["name"], base_storage_dir)
        except ValueError as ve:
            logger.error(
                f"Skipping dataset '{data_dict['name']}' due to processing error: {ve}"
            )
        except FileNotFoundError as fnf:
            logger.error(
                f"File not found while processing dataset '{data_dict['name']}': {fnf}"
            )
        except Exception as e:
            logger.error(
                f"An unexpected error occurred while processing dataset '{data_dict['name']}': {e}",
                exc_info=True,
            )
        finally:
            logger.info("-" * 30)

    logger.info("All datasets processed.")


if __name__ == "__main__":
    main()