"""Utility to load and process the dataset.

This module is intended to ease the use of the preprocessing pipeline
that was developed in the explore.ipynb notebook.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.utils import shuffle
from tensorflow.keras.utils import PyDataset

logger = logging.getLogger()
logger.setLevel("DEBUG")

DATASET_HOME = Path.cwd() /  "data"
TRAIN_IMAGES_PATH = DATASET_HOME / "train-image" / "image"
METADATA_PATH = DATASET_HOME / "train-metadata.csv"

MOST_COMMON_SHAPE = (133,133)


class SkinCancerDataset(PyDataset):
    """Generator for dynamically loading the dataset"""

    def __init__(self, x_set, y_set, batch_size, **kwargs):
        super().__init__(**kwargs)
        self.x = x_set
        self.y = y_set
        self.batch_size = batch_size
        self.class_weights = self.calculate_class_weights()
        self.weight_mapper = self.create_weight_mapper()

    def calculate_class_weights(self) -> dict[int,int]:
        """Create a dictionary for the class weights"""
        half_count = len(self.y) / 2
        positive_samples = self.y.sum()
        negative_samples = len(self.y) - positive_samples
        return {
            0: half_count / negative_samples,
            1: half_count / positive_samples
        }
    
    def create_weight_mapper(self) -> np.ndarray:
        """Create a mapper that returns the weight corresponding to the given label"""
        return np.vectorize(self.class_weights.get, otypes=[float])

    def __len__(self) -> int:
        # Number of batches
        return math.ceil(len(self.y) / self.batch_size)

    def _load_image_batch(self, batch: list[str]) -> np.ndarray:
        X = []
        for filename in batch:
            image = Image.open(filename).resize(MOST_COMMON_SHAPE)
            np_scaled = np.array(image) / 255
            X.append(np_scaled)
            image.close()
        return np.array(X)

    def __getitem__(self, idx: int) -> np.ndarray:
        start_idx = self.batch_size * idx
        end_idx = min(self.batch_size + start_idx, len(self.x))

        batch_x = self.x[start_idx:end_idx]
        batch_y = self.y[start_idx:end_idx]

        X = self._load_image_batch(batch_x)
        return X, batch_y, self.weight_mapper(batch_y)
        


def create_dataset(metadata, batch_size:int=50) -> SkinCancerDataset:
    """Creates a SkinCancerDataset from the given metadata
    
    Args:
        metadata (pd.DataFrame): Corresponding metadata for the images to be loaded

    Returns:
        SkinCancerDataset: A dataset generator for the images in the metadata
    """
    filenames = [str(TRAIN_IMAGES_PATH / (id + ".jpg")) for id in metadata["isic_id"]]
    labels = metadata["target"].to_numpy()

    filenames, labels = shuffle(filenames, labels)
    return SkinCancerDataset(filenames, labels, batch_size)


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
    metadata = load_metadata()
    X, Y, metadata = _load_images(metadata, load_size=load_size)
    metadata, pipeline = _preprocess_metadata(metadata)

    return X, metadata, Y, pipeline


def load_metadata() -> pd.DataFrame:
    """Load metadata from csv and select relevant columns"""
    metadata = pd.read_csv(METADATA_PATH, dtype={"target": "int8", "age_approx": "int8"})
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


def upsample_metadata(metadata: pd.DataFrame, upsample_factor: int) -> pd.DataFrame:
    """Upsample the positive samples in the metadata by the upsample_factor"""
    metadata["upsampled"] = 0
    positive_samples = metadata[metadata["target"] == 1]
    positive_samples["upsamples"] = 1
    for i in range(upsample_factor):
        metadata = pd.concat([metadata, positive_samples])
    return metadata



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
            image = Image.open(file_path)
            image.thumbnail(MOST_COMMON_SHAPE)
            np_image = np.array(image) / 255
            X.append(np_image)
            image.close()
            del image
        else:
            missing_ids.append(index)
            logger.warning("File %s.jpg does not exist.", filename)
    logger.info("Number of images loaded: %s", len(X))

    metadata.drop(missing_ids)
    Y = metadata["target"].to_numpy()
    value_counts = metadata["target"].map({0: "Benign", 1: "Malignant"}).value_counts()
    logger.info(value_counts)

    return X, Y, metadata


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

