"""Utility to load and process the dataset.

This module is intended to ease the use of the preprocessing pipeline
that was developed in the explore.ipynb notebook.
"""

from __future__ import annotations

import logging
import math
import random
import zipfile
from pathlib import Path
from typing import Any

import gdown
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
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


def download_data():
    """Utility function to download the raw dataset"""
    if(not Path("data").exists()):
        # extract zip to the data dir
        gdown.download("https://drive.google.com/uc?id=13z3O9BI082DFGs8aSaCAzWDbYCs_ZLxT", "resources.zip", quiet=False)
        with zipfile.ZipFile("resources.zip", 'r') as zip_ref:
            zip_ref.extractall("data")


# List of functions to augment images
image_augmenters = [
    ImageOps.flip,
    ImageOps.mirror,
    lambda image: image.crop((10, 10,123,123)).resize(MOST_COMMON_SHAPE)
]


class SkinCancerDataset(PyDataset):
    """Generator for dynamically loading the dataset"""

    def __init__(self, file_info: dict, labels:int, batch_size:int, **kwargs):
        super().__init__(**kwargs)
        self.x = file_info
        self.y = labels
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
        for file_info in batch:
            path = file_info["filepath"]
            upsampled = file_info["upsampled"]
            image = Image.open(path).resize(MOST_COMMON_SHAPE)
            if upsampled:
                image = self._augment_image(image)
            np_scaled = np.array(image) / 255
            X.append(np_scaled)
            image.close()
        return np.array(X)
    
    def _augment_image(self, image: Image):
        """Applies either one or two transformations from image_augmenters"""
        nb_transforms = random.randint(1,2)
        rand_indices = random.sample(range(0, len(image_augmenters) - 1), nb_transforms)
        for idx in rand_indices:
            image = image_augmenters[idx](image)
        return image

    def __getitem__(self, idx: int) -> np.ndarray:
        start_idx = self.batch_size * idx
        end_idx = min(self.batch_size + start_idx, len(self.x))

        batch_x = self.x[start_idx:end_idx]
        batch_y = self.y[start_idx:end_idx]

        X = self._load_image_batch(batch_x)
        return X, batch_y, self.weight_mapper(batch_y)
        

class SkinCancerReconstructionDataset(PyDataset):
    """Generator for dynamically loading the dataset for training an autoencoder
    
    The target in this dataset will be the images themself, meaning
    that the __getitem__ method returns a batch of images two times
    in a tuple
    """
    
    def __init__(self, filepath: dict, batch_size:int, **kwargs):
        super().__init__(**kwargs)
        self.x = filepath
        self.batch_size = batch_size

    def __len__(self) -> int:
        # Number of batches
        return math.ceil(len(self.x) / self.batch_size)

    def _load_image_batch(self, batch: list[str]) -> np.ndarray:
        X = []
        for filepath in batch:
            image = Image.open(filepath).resize(MOST_COMMON_SHAPE)
            np_scaled = np.array(image) / 255
            X.append(np_scaled)
            image.close()
        return np.array(X)

    def __getitem__(self, idx:int) -> np.ndarray:
        start_idx = self.batch_size * idx
        end_idx = min(self.batch_size + start_idx, len(self.x))

        batch_x = self.x[start_idx:end_idx]
        X = self._load_image_batch(batch_x)
        return X, X


def create_dataset(metadata:pd.DataFrame, batch_size:int=50, workers:int=6) -> SkinCancerDataset:
    """Creates a SkinCancerDataset from the given metadata
    
    Args:
        metadata (pd.DataFrame): Corresponding metadata for the images to be loaded

    Returns:
        SkinCancerDataset: A dataset generator that yields (image, label, class_weigth) batches from the metadata
    """
    # Extend the id to the complete path
    metadata["filepath"] = metadata["isic_id"].apply(lambda id: str(TRAIN_IMAGES_PATH / (id + ".jpg")))
    file_info = metadata[["filepath", "upsampled"]].to_dict(orient="records")
    labels = metadata["target"].to_numpy()

    file_info, labels = shuffle(file_info, labels)
    return SkinCancerDataset(file_info, labels, batch_size, workers=workers, use_multiprocessing=True)


def create_reconstruction_dataset(metadata:pd.DataFrame, batch_size:int=50, workers:int=6) -> SkinCancerReconstructionDataset:
    """Creates a SkinCancerReconstructionDataset from the given metadata
    
    Args:
        metadata (pd.DataFrame): Corresponding metadata for the images to be loaded

    Returns:
        SkinCancerReconstructionDataset: A dataset generator that yields batches of (image, image) from the metadata
    """
    filepath = metadata["isic_id"].apply(lambda id: str(TRAIN_IMAGES_PATH / (id + ".jpg"))).to_list()
    return SkinCancerReconstructionDataset(filepath, batch_size, workers=workers, use_multiprocessing=True)


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


def load_metadata(sample_fraction: float = 1.0) -> pd.DataFrame:
    """Load metadata from csv and select relevant columns"""
    metadata = pd.read_csv(METADATA_PATH, dtype={"target": "int8", "age_approx": "Int8"})
    if sample_fraction < 1.0:
        metadata = metadata.sample(frac=sample_fraction, random_state=42)
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
    positive_samples["upsampled"] = 1
    for _ in range(upsample_factor):
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

