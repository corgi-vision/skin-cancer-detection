from __future__ import annotations

import math
import random
from pathlib import Path

import wandb
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Cropping2D, Flatten
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.utils import PyDataset


MOST_COMMON_SHAPE = (133,133)
INPUT_SHAPE = (133,133,3)
LOCAL_REGISTRY = "local_model_registry"


# List of functions to augment images
image_augmenters = [
    ImageOps.flip,
    ImageOps.mirror,
    lambda image: image.crop((10, 10,123,123)).resize(MOST_COMMON_SHAPE)
]


def _augment_image(image: Image):
    """Applies either one or two transformations from image_augmenters"""
    nb_transforms = random.randint(1,2)
    rand_indices = random.sample(range(0, len(image_augmenters) - 1), nb_transforms)
    for idx in rand_indices:
        image = image_augmenters[idx](image)
    return image


# The Autoencoder class is defined in the class to make model loading possible
@register_keras_serializable()
class Autoencoder(Model):
    """Autoencoder to create an embedding for the images"""

    def __init__(self, **kwargs):
        super(Autoencoder, self).__init__(**kwargs)
        self.encoder = Sequential([
            Input(INPUT_SHAPE),
            Conv2D(32, 5, activation="relu", padding="same", strides=2),
            Conv2D(16, 3, activation="relu", padding="same", strides=2),
            Conv2D(1, 3, activation="relu", padding="same", strides=2),
        ])
        self.decoder = Sequential([
            Conv2DTranspose(8, 3, strides=2, padding="same", activation="relu"),
            Conv2DTranspose(16, 3, strides=2, padding="same", activation="relu"),
            Conv2DTranspose(32, 5, strides=2, padding="same", activation="relu"),
            Conv2D(3, 3, activation="sigmoid", padding="same"),
            Cropping2D(((2,1), (2,1)))
        ])
    
    def get_config(self):
        super().get_config()

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


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
                image = _augment_image(image)
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
    

class SkinCancerConcatDataset(PyDataset):
    """Generator for dynamically loading the dataset"""

    def __init__(self, file_info: dict, metadata, labels:int, batch_size:int, **kwargs):
        super().__init__(**kwargs)
        self.x = [file_info,metadata]
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
                image = _augment_image(image)
            np_scaled = np.array(image) / 255
            X.append(np_scaled)
            image.close()
        return np.array(X)

    def __getitem__(self, idx: int) -> np.ndarray:
        start_idx = self.batch_size * idx
        end_idx = min(self.batch_size + start_idx, len(self.y))

        batch_image = self.x[0][start_idx:end_idx]
        batch_image = self._load_image_batch(batch_image)
        batch_metadata = self.x[1][start_idx:end_idx]
        batch_y = self.y[start_idx:end_idx]

      
        
        X = (batch_image,batch_metadata)
        #X = self._load_image_batch(batch_x)
        return (batch_image,batch_metadata), batch_y, self.weight_mapper(batch_y)
        

class SkinCancerReconstructionDataset(PyDataset):
    """Generator for dynamically loading the dataset for training an autoencoder
    
    The target in this dataset will be the images themself, meaning
    that the __getitem__ method returns a batch of images two times
    in a tuple
    """
    
    def __init__(self, filepath: list, batch_size:int, **kwargs):
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
    

class SkinCancerEncodedDataset(PyDataset):
    """
    Generator for dynamically loading the dataset encoded with an autoencoder
    and concatenated with the processed metadata
    """

    def __init__(self, metadata:np.ndarray, file_info:dict, labels:np.ndarray, encoder: dict, batch_size:int, **kwargs):
        super().__init__(**kwargs)
        self.file_info = file_info
        self.metadata = metadata
        self.labels = labels
        self.batch_size = batch_size
        self.class_weights = self.calculate_class_weights()
        self.weigth_mapper = self.create_weight_mapper()
        self.model = self._load_model(encoder)

    def __len__(self) -> int:
        # Number of batches
        return math.ceil(len(self.labels) / self.batch_size)
    
    def __getitem__(self, idx:int) -> np.ndarray:
        start_idx = self.batch_size * idx
        end_idx = min(self.batch_size + start_idx, len(self.labels))

        batch_files = self.file_info[start_idx:end_idx]
        batch_metadata = self.metadata[start_idx:end_idx]
        batch_Y = self.labels[start_idx:end_idx]
        batch_X = self._load_encoded_batch(batch_files, batch_metadata)

        return batch_X, batch_Y, self.weigth_mapper(batch_Y)
    
    def calculate_class_weights(self) -> dict[int,int]:
        """Create a dictionary for the class weights"""
        half_count = len(self.labels) / 2
        positive_samples = self.labels.sum()
        negative_samples = len(self.labels) - positive_samples
        return {
            0: half_count / negative_samples,
            1: half_count / positive_samples
        }

    def create_weight_mapper(self) -> np.ndarray:
        """Create a mapper that returns the weight corresponding to the given label"""
        return np.vectorize(self.class_weights.get, otypes=[float])


    def _load_model(self, encoder:wandb.Artifact):
        model_path = Path(LOCAL_REGISTRY) / encoder["name"]
        # Check if model is already downloaded
        if not model_path.exists():
            print("Downloading artifact")
            encoder["artifact"].download(LOCAL_REGISTRY)

        return load_model(model_path, custom_objects={"Autoencoder": Autoencoder})

    def _load_encoded_batch(self, batch_files:dict, batch_metadata:np.ndarray):
        X = []
        # Load and augment images
        for file_info in batch_files:
            path = file_info["filepath"]
            upsampled = file_info["upsampled"]
            image = Image.open(path).resize(MOST_COMMON_SHAPE)
            if upsampled:
                image = _augment_image(image)
            np_scaled = np.array(image) / 255
            X.append(np_scaled)
            image.close()
        # Encode the image and concatenate with metadata
        encoded_image = Flatten()(self.model.encoder(np.array(X)))
        return np.concatenate([encoded_image, batch_metadata], axis=1)