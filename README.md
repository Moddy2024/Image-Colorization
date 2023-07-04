# Image-Colorization


My implementation of ["Image-to-Image Translation with Conditional Adversarial Networks"](https://arxiv.org/pdf/1611.07004v3.pdf) is based on the paper by Isola et al. The generator architecture used in the paper is an UNET architecture, the results produced from it for Image Colorization were not satisfactory. So, I created a different Generator architecture where the encoder was influenced by Resnet but the Decoder is the same as paper as this produces better visually appealing images. I even trained the Generator on a NOGan training first which was inspired from DeOldify.

The modified generator architecture comprises multiple components organized in a sequential manner. The Encoder stage involves down-sampling the input image using a convolutional layer with appropriate parameters. This step is followed by a series of residual blocks, each consisting of convolutional layers with batch normalization and leaky ReLU activation functions. Subsequently, a series of residual blocks further refine the extracted features, ensuring the preservation of important image details while facilitating the learning of intricate patterns. Additionally, skip connections are incorporated to facilitate the flow of information from earlier layers to later layers, enhancing gradient flow and enabling the model to capture fine-grained details.
The Decoder layers of the generator architecture involve an upsampling process and a final convolutional layer with a transposed convolution operation. Critically, at each upsampling step, the upsampled feature maps are concatenated with the corresponding feature maps from the skip connections. This concatenation operation enables the fusion of high-level semantic information from the Encoder with detailed information captured at earlier stages. The concatenation operation was followed as it was in the paper. The output of the generator is the synthesized ab channel of LAB image that is further converted to RGB for the colorized Image.
![]()

The Discriminator implements a patch discriminator which is exactly the same as discussed in the paper, which is a key component in generative adversarial networks (GANs) for image analysis tasks. The patch discriminator architecture is specifically designed to discern the authenticity of local image patches rather than the entire image. This approach facilitates fine-grained discrimination and enables the capture of intricate features within an image. The architecture of the patch discriminator is composed of a sequence of convolutional layers with progressively reduced spatial dimensions. Following each convolutional layer, batch normalization and LeakyReLU activation functions are applied. Batch normalization aids in stabilizing the training process by normalizing the layer inputs, while LeakyReLU activation introduces non-linearity and enhances the model's ability to capture complex patterns. The final layer of the discriminator produces a single output channel, representing the probability of the input patch being real or fake. This output probability is then utilized in tandem with the generator to train the discriminator within the GAN framework.
![](https://github.com/Moddy2024/Image-Colorization/blob/main/discrminator-diagram/discriminator-image.png)

The image of the architecture of the discrminator has been created using [PlotNeuralNet](https://github.com/HarisIqbal88/PlotNeuralNet).
 
This implementation is in PyTorch framework. I have provided a user-friendly interface for users to upload their images, and generate colorized images.

The model has been trained for 2,000,000 iterations on a Kaggle Notebook with GPU (Nvidia Tesla P100 16GB) which took 80 hours approximately.

# Dependencies
* [PyTorch](https://pytorch.org/)
* [Matplotlib](https://matplotlib.org/)
* [PIL](https://pypi.org/project/Pillow/)
* [Numpy](https://numpy.org/)
* [OS](https://docs.python.org/3/library/os.html)
* [Scikit-Image](https://scikit-image.org/)
* [tqdm](https://tqdm.github.io/)
* [Torchvision](https://pytorch.org/vision/stable/index.html)
* [torchinfo](https://github.com/TylerYep/torchinfo)
* [Pathlib](https://docs.python.org/3/library/pathlib.html)

Once you have these dependencies installed, you can clone the Image Colorization repository from GitHub:
```bash
https://github.com/Moddy2024/Image-Colorization.git
```
# Key Directories and Files
* [training-gan.ipynb](https://github.com/Moddy2024/Image-Colorization/blob/main/training-gan.ipynb) - In this Jupyter Notebook, you can find a comprehensive walkthrough of the data pipeline for Image-Colorization, which includes steps for downloading the dataset, preprocessing the data, details on the various data transformations and data loader creation steps,  along with visualizations of the data before and after transformations and moving the preprocessed data to the GPU for model training. Moreover, the notebook delves into the meticulous implementation of the model architecture for both the generator and the discriminator, accompanied by the whole training process.
* []() - 
* [prediction.ipynb]() - This notebook demonstrates how to perform style transfer on images using the pre-trained model.
* [results](https://github.com/Moddy2024/Image-Colorization/tree/main/results) - This directory contains the colorized images from the  test images that ha.
* [test-images](https://github.com/Moddy2024/Image-Colorization/tree/main/test-images) - This directory contains a collection of black and white old images sourced randomly from the internet, which are intended to be used for testing and evaluation purposes.
* [discrminator-diagram](https://github.com/Moddy2024/Image-Colorization/tree/main/discrminator-diagram) - This directory contains a collection of content images sourced randomly from the internet, which are intended to be used for testing and evaluation purposes.
# Dataset
The [ImageNet](https://www.image-net.org/about.php) Dataset has been downloaded and extracted from the Kaggle Competition [ImageNet Object Localization Challenge](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data). It can also be retrieved from it's official website [here](https://www.image-net.org/download.php) in order to download from this website you will have to sign up there. Yoi can either used the dataset in Kaggle itself but if you want to download to use it somewhere else then first you need to extract your [API Token](https://www.kaggle.com/discussions/general/371462#2060661) from the Kaggle account only then you will be able to download dataset from Kaggle to anywhere. The official instructions on how to use the [KAGGLE API](https://github.com/Kaggle/kaggle-api).
```bash
!ls -lha /home/ec2-user/SageMaker/kaggle.json
!pip install -q kaggle
!mkdir -p ~/.kaggle #Create the directory
!cp kaggle.json ~/.kaggle/
!chmod 600 /home/ec2-user/SageMaker/kaggle.json

!kaggle competitions download -f train.zip -p '/home/ec2-user/SageMaker' -o imageNet-object-localization-challenge
local_zip = '/home/ec2-user/SageMaker/train.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
!mkdir /home/ec2-user/SageMaker/imagenet-data
zip_ref.extractall('/home/ec2-user/SageMaker/imagenet-data')
zip_ref.close()
os.remove(local_zip)
print('The number of images present in Imagenet dataset are:',len(os.listdir('/home/ec2-user/SageMaker/train')))
```

# Results
  ![](https://github.com/Moddy2024/Image-Colorization/blob/main/results/727.png)
  ![](https://github.com/Moddy2024/Image-Colorization/blob/main/results/7752.png)
  ![](https://github.com/Moddy2024/Image-Colorization/blob/main/results/824.png)
  ![](https://github.com/Moddy2024/Image-Colorization/blob/main/results/856.png)
  ![](https://github.com/Moddy2024/Image-Colorization/blob/main/results/858.png)
  ![](https://github.com/Moddy2024/Image-Colorization/blob/main/results/859.png)
  ![]()
 # Citation
```bash
@inproceedings{isola2018pix2pix,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A.},
  booktitle={CVPR},
  year={2018}
}
```
