# Skin cancer detection with deep neural network

## CorgiVision group project

### Participants:

* Barbara Kéri - AR5KHR
* Benjámin Csontó - JTB4Y1
* Sámuel Csányi - I7ULKV

### Kaggle ISIC 2024 challange 

In this project, we will develop image-based algorithms to identify histologically confirmed skin cancer cases with single-lesion crops from 3D total body photos (TBP). The image quality resembles close-up smartphone photos, which are regularly submitted for telehealth purposes. The dataset consists of 401059 images and among their corresponding labels (positive or negative), various metadata about the patient and the leisure. We've narrowed down the features to the ones that should be useful in machine learning applications. They are listed in [info.md](info.md)

### Data preparation

The scripts for loading, visualizing and preprocessing the data are in the following notebook: [explore.ipynb](explore.ipynb)

The preprocessing pipeline has been exported to [preprocessing.py](preprocessing.py) and augmented with the following functionality:
[data_loading.py](data_loading.py) contains the generator classes to dynamically load the dataset.

#### Autoencoder
See [training_autoencoder.ipynb](training_autoencoder.ipynb) <br>
In this notebook an autoencoder is trained using the `SkinCancerReconstructionDataset` generator class, and as a result we obtain an encoder and a decoder for the images.
With the encoder we are able to create an embedding, we can concatenate the metadata to. With this concatenated dataset, we will train another model that receives both
the images and the metadata as its input.


#### Inception-ResNet
See [training_resnet.ipynb](training_resnet.ipynb) <br>
In this notebook, we are utilizing transfer learning with a pretrained Inception-ResNet model, which is an extension of the Inception family of architectures incorporating residual connections. 

Additional layers are appended to the pretrained model, with most of the pretrained layers remaining frozen during training.
Finally, we will compare the evaluation results of both architectures to determine which approach best suits the dataset and task.
