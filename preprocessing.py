"""Utility to load and process the dataset.

This module is intended to ease the use of the preprocessing pipeline
that was developed in the explore.ipynb notebook.
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from typing import Any

from data_loading import SkinCancerDataset, SkinCancerReconstructionDataset, SkinCancerEncodedDataset, SkinCancerConcatDataset

import gdown
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.utils import shuffle

logger = logging.getLogger()
logger.setLevel("DEBUG")

DATASET_HOME = Path.cwd().parent /  "data"
TRAIN_IMAGES_PATH = DATASET_HOME / "train-image" / "image"
METADATA_PATH = DATASET_HOME / "train-metadata.csv"



def download_data():
    """Utility function to download the raw dataset"""
    if(not Path(DATASET_HOME).exists()):
        # extract zip to the data dir
        gdown.download("https://drive.google.com/uc?id=13z3O9BI082DFGs8aSaCAzWDbYCs_ZLxT", "resources.zip", quiet=False)
        with zipfile.ZipFile("resources.zip", 'r') as zip_ref:
            zip_ref.extractall(DATASET_HOME)


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

def create_concat_dataset(metadata:pd.DataFrame, preproc_meta, batch_size:int=50, workers:int=6) -> SkinCancerDataset:
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
    file_info, preproc_meta, labels = shuffle(file_info, preproc_meta, labels)
    return SkinCancerConcatDataset(file_info, preproc_meta, labels, batch_size, workers=workers, use_multiprocessing=True)


def create_reconstruction_dataset(metadata:pd.DataFrame, batch_size:int=50, workers:int=6) -> SkinCancerReconstructionDataset:
    """Creates a SkinCancerReconstructionDataset from the given metadata
    
    Args:
        metadata (pd.DataFrame): Corresponding metadata for the images to be loaded

    Returns:
        SkinCancerReconstructionDataset: A dataset generator that yields batches of (image, image) from the metadata
    """
    filepath = metadata["isic_id"].apply(lambda id: str(TRAIN_IMAGES_PATH / (id + ".jpg"))).to_list()
    return SkinCancerReconstructionDataset(filepath, batch_size, workers=workers, use_multiprocessing=True)


def create_encoded_dataset(metadata:pd.DataFrame, encoder:dict, batch_size:int=50, workers:int=6) -> SkinCancerEncodedDataset:
    """Creates a SkinCancerEncodedDataset from the given metadata
    
    Args:
        metadata (pd.DataFrame): Corresponding metadata for the images to be loaded
        encoder (dict): Dictionary, that has a key for the model's name and one for the artifact
    
    Returns:
        SkinCancerEncodedDataset: A dataset generator that yields (flattened image + metadata, label, weight)
    """

    metadata["filepath"] = metadata["isic_id"].apply(lambda id: str(TRAIN_IMAGES_PATH / (id + ".jpg")))
    file_info = metadata[["filepath", "upsampled"]].to_dict(orient="records")
    labels = metadata["target"].to_numpy()
    processed_metadata, pipeline = _preprocess_metadata(metadata)   # TODO: save pipeline somehow
    return SkinCancerEncodedDataset(processed_metadata, file_info, labels, encoder, batch_size)



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

import os

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
            image.thumbnail((133,133))
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