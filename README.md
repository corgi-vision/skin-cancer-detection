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

**Dynamic data loading**

The dataset in processed for is too big to fit into memory *133 * 133 * 3 * float32*, even if we would use a data type with smaller precision. Our solution is to use tf.keras.utils.PyDataset as a base class for our dataset, and let it handle the dynamic loading of the data. The `create_dataset()` utility function uses this class to create a dataset object from the metadata that it receives.
The other dataset generator is the `SkinCancerReconstructionDataset` which generates batches where the taget is the same as the input. This dataset is used for training an autoencoder. It has a utility function as well: `create_reconstruction_dataset()`.

**Balancing the class samples**

Positive samples are heavily under-represented, which needs to be balanced out. We use the following techniques to compensate:
* **Upsampling**<br>
    Datapoints which belong to the positive samples are added to the dataset multiple times. This is indicated by the `upscale_factor` <br>
    parameter when calling the `upscale_metata()` method.
* **Data augmenting**<br>
    To make the upsampled images more unique, some image augmentation techniques are applied. In particular horizontal and vertical mirroring <br>
    and cropping then rescaling the images. Either one or two methods are applied randomly.
* **Sample weights**<br>
    For each sample the loss function is evaluated using a corresponding weight, <br>
    which is higher for the positive samples. We use to following formula: $c_d / (2 * c_s)$, <br>
    where $c_d$ is the count of all samples and $c_s$ is the count of samples for a given class of labels.


### Model training
Multiple architectures are created, trained and evaluated in order to explore different possibilities and find the best results.


#### Autoencoder
See [training_autoencoder.ipynb](training_autoencoder.ipynb) <br>
In this notebook an autoencoder is trained using the `SkinCancerReconstructionDataset` generator class, and as a result we obtain an encoder and a decoder for the images.
With the encoder we are able to create an embedding, we can concatenate the metadata to. With this concatenated dataset, we will train another model that receives both
the images and the metadata as its input.


#### ResNet
See [training_resnet.ipynb](training_resnet.ipynb) <br>
In this notebook, we are utilizing transfer learning with a pretrained Inception-ResNet model, which is an extension of the Inception family of architectures incorporating residual connections. 

Additional layers are appended to the pretrained model, with most of the pretrained layers remaining frozen during training.
Finally, we will compare the evaluation results of both architectures to determine which approach best suits the dataset and task.
