"""Utility to load and process the dataset.

This module is intended to ease the use of the preprocessing pipeline
that was developed in the explore.ipynb notebook.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

logger = logging.getLogger()
DATASET_HOME = Path.cwd() /  "data"
TRAIN_IMAGES_PATH = DATASET_HOME / "train-image" / "image"
METADATA_PATH = DATASET_HOME / "train-metadata.csv"


def load_prepared_datasets(load_size:float=1) -> tuple[Any]:
    """Load and preprocess the dataset.

    Split the data into features and labels
    and return relevant metadata and preprocessing pipeline.

    Args:
        load_size (float): How much of the whole dataset will be loaded.

    Returns:
        tuple of:
            - X (array-like): Scaled images.
            - Y (array-like): Labels.
            - metadata (array-like): Processed metadata, corresponding to the images
            - pipeline (Pipeline): Preprocessing pipeline for transforming the metadata.

    """
    metadata = _load_metadata()
    X, Y, metadata = _load_images(metadata, load_size=load_size)

    X_scaled = _rescale_images(X)
    metadata, pipeline = _preprocess_metadata(metadata)

    return X_scaled, metadata, Y, pipeline


def _load_metadata() -> pd.DataFrame:
    metadata = pd.read_csv(METADATA_PATH)
    return metadata[
        [
            "isic_id",
            "target",
            "age_approx",
            "sex",
            "tbp_lv_areaMM2",
            "tbp_lv_area_perim_ratio",
            "tbp_lv_color_std_mean",
            "tbp_lv_deltaLBnorm",
            "tbp_lv_location",
            "tbp_lv_minorAxisMM",
            "tbp_lv_nevi_confidence",
            "tbp_lv_norm_border",
            "tbp_lv_norm_color",
            "tbp_lv_perimeterMM",
            "tbp_lv_radial_color_std_max",
            "tbp_lv_symm_2axis",
            "tbp_lv_symm_2axis_angle",
        ]
    ]


def _load_images(metadata:pd.DataFrame, load_size:float=1) -> tuple[Any]:
    load_count = int(metadata.shape[0] * load_size)
    logger.info("Number of images to be loaded: %s", load_count)

    metadata = metadata.iloc[:load_count]
    missing_ids=[]
    X = []

    for index, row in metadata.iterrows():
        filename = row["isic_id"]
        file_path = TRAIN_IMAGES_PATH / f"{filename}.jpg"

        if file_path.exists():
            X.append(Image.open(file_path))
        else:
            missing_ids.append(index)
            logger.warning("File %s.jpg does not exist.", filename)
    logger.info("Number of images loaded: %s", len(X))

    metadata.drop(missing_ids)
    Y = metadata["target"].to_numpy()
    value_counts = metadata["target"].map({0: "Benign", 1: "Malignant"}).value_counts()
    logger.info(value_counts)

    return X, Y, metadata


def _rescale_images(X:list) -> np.ndarray:
    sizes = [im.size for im in X]
    not_square = list(filter(lambda s: s[0] != s[1] ,sizes))
    logger.warning("%s images are not n by n.", len(not_square))

    values, counts = np.unique(sizes, return_counts=True)
    most_frequent = values[np.argmax(counts)]
    logger.info("Most frequent size: %s", most_frequent)

    resized_images = []
    for im in X:
        scaled = im.resize([most_frequent, most_frequent])
        resized_images.append(scaled)
        im.close()
    X.clear()

    # Mapping to a np array and scaling the images to the [0,1] interval
    return np.array(resized_images) / 255


def _preprocess_metadata(metadata:pd.DataFrame) -> tuple[Any]:
    numerical_features = [
        "age_approx",
        "tbp_lv_areaMM2",
        "tbp_lv_area_perim_ratio",
        "tbp_lv_color_std_mean",
        "tbp_lv_deltaLBnorm",
        "tbp_lv_minorAxisMM",
        "tbp_lv_nevi_confidence",
        "tbp_lv_norm_border",
        "tbp_lv_norm_color",
        "tbp_lv_perimeterMM",
        "tbp_lv_radial_color_std_max",
        "tbp_lv_symm_2axis",
        "tbp_lv_symm_2axis_angle",
    ]
    binary_features = ["sex"]
    categorical_features = ["tbp_lv_location"]

    # Create preprocessing pipelines for reusability
    # Fill in missing features, standardize numerical values and encode categorical ones
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    binary_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder()),
    ])

    preprocessing_pipeline = ColumnTransformer([
        ("numerical", numerical_pipeline, numerical_features),
        ("binary", binary_pipeline, binary_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    metadata_processed = preprocessing_pipeline.fit_transform(metadata)
    return metadata_processed, preprocessing_pipeline

