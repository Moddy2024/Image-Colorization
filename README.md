# Image-Colorization


My implementation of ["Image-to-Image Translation with Conditional Adversarial Networks"](https://arxiv.org/pdf/1611.07004v3.pdf) is based on the paper by Isola et al. The generator architecture used in the paper is an UNET architecture, the results produced from it for Image Colorization were not satisfactory. So, I created a different Generator architecture where the encoder was influenced by Resnet but the Decoder is the same as paper as this produces better visually appealing images. I even trained the Generator on a NOGan training first which was inspired from [DeOldify](https://deoldify.ai/).

The modified generator architecture comprises multiple components organized in a sequential manner. The Encoder stage involves down-sampling the input image using a convolutional layer with appropriate parameters. This step is followed by a series of residual blocks, each consisting of convolutional layers with batch normalization and leaky ReLU activation functions. Subsequently, a series of residual blocks further refine the extracted features, ensuring the preservation of important image details while facilitating the learning of intricate patterns. Additionally, skip connections are incorporated to facilitate the flow of information from earlier layers to later layers, enhancing gradient flow and enabling the model to capture fine-grained details.
The Decoder layers of the generator architecture involve an upsampling process and a final convolutional layer with a transposed convolution operation. Critically, at each upsampling step, the upsampled feature maps are concatenated with the corresponding feature maps from the skip connections. This concatenation operation enables the fusion of high-level semantic information from the Encoder with detailed information captured at earlier stages. The concatenation operation was followed as it was in the paper. The output of the generator is the synthesized ab channel of LAB image that is further converted to RGB for the colorized Image.
![]()

The Discriminator implements a patch discriminator which is exactly the same as discussed in the paper, which is a key component in generative adversarial networks (GANs) for image analysis tasks. The patch discriminator architecture is specifically designed to discern the authenticity of local image patches rather than the entire image. This approach facilitates fine-grained discrimination and enables the capture of intricate features within an image. The architecture of the patch discriminator is composed of a sequence of convolutional layers with progressively reduced spatial dimensions. Following each convolutional layer, batch normalization and LeakyReLU activation functions are applied. Batch normalization aids in stabilizing the training process by normalizing the layer inputs, while LeakyReLU activation introduces non-linearity and enhances the model's ability to capture complex patterns. The final layer of the discriminator produces a single output channel, representing the probability of the input patch being real or fake. This output probability is then utilized in tandem with the generator to train the discriminator within the GAN framework.
![](https://github.com/Moddy2024/Image-Colorization/blob/main/discrminator-diagram/discriminator-image.png)

The image of the architecture of the discrminator has been created using [PlotNeuralNet](https://github.com/HarisIqbal88/PlotNeuralNet).
 
This implementation is in PyTorch framework. I have provided a user-friendly interface for users to upload their images, and generate colorized images.

The model has been trained for 2,000,000 iterations on [1.2 million images](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data) in a Kaggle Notebook with GPU (Nvidia Tesla P100 16GB) which took 80 hours approximately.

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
* [training-nogan.ipynb](https://github.com/Moddy2024/Image-Colorization/blob/main/training-nogan.ipynb) - In this Jupyter Notebook, you can find a comprehensive walkthrough of the data pipeline for Image-Colorization non-gan training, which includes steps for downloading the dataset, preprocessing the data, details on the various data transformations and data loader creation steps,  along with visualizations of the data before and after transformations and moving the preprocessed data to the GPU for model training. Moreover, the notebook delves into the meticulous implementation of the Nogan training process which is necessary for accurate color predictions.
* [prediction.ipynb](https://github.com/Moddy2024/Image-Colorization/blob/main/prediction.ipynb) - This notebook demonstrates how to perform colorization, saving the image and creating the side by images of greyscale image and the color image using the pre-trained model.
* [results](https://github.com/Moddy2024/Image-Colorization/tree/main/results) - This directory contains the colorized images of the  test images.
* [test-images](https://github.com/Moddy2024/Image-Colorization/tree/main/test-images) - This directory contains a collection of black and white old images sourced randomly from the internet, which are intended to be used for testing and evaluation purposes.
* [discrminator-diagram](https://github.com/Moddy2024/Image-Colorization/tree/main/discrminator-diagram) - This directory contains the codes necessary for creating the diagram of the discrminator architecture using [PlotNeuralNet](https://github.com/HarisIqbal88/PlotNeuralNet) .
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
# Training
The training process is divided into two parts: NOGAN training influenced by [DeOldify](https://github.com/jantic/DeOldify#what-is-nogan) and normal GAN training as prescribed by the [paper](https://arxiv.org/pdf/1611.07004v3.pdf). The reason for doing NoGan Training is that it to close the gap on realism and achieve full realistic colorization capabilities. As this is a NOGAN training phase, only generator is used here the discriminator is not needed, and the loss is calculated as follows: First, the L1 loss is calculated between the ab channel generated by the generator and the original ab channel. This loss is then multiplied by a lambda value of 100 to control its effect on the generator. The resulting loss is added to another loss called the feature loss. Feature Loss is calculated using pretrained VGG19 with batchnorm. The VGG19 model's only CNN layers are used they are sliced and divided into three parts to extract features from different levels. The ab channel is concatenated with the L channel, and this composite image is inputted into the pretrained network. The feature loss is calculated by comparing the extracted feature representations from the slices of the real image's LAB channel and the fake image's LAB channel, the feature loss is calculated as the element-wise absolute difference (L1 loss) between the feature slices of the fake and real images. The feature loss is computed for each layer of the VGG19 network, and the losses from each layer are weighted using predefined weights(which can be altered). The final feature loss is the sum of the weighted losses from each layer. The feature loss is added to the previously calculated L1 loss, and the weights of the model are updated accordingly. This combined loss, consisting of the L1 loss and the feature loss, guides the generator in producing visually appealing and color-accurate results. The generator should be trained for about 2-3 epochs on NOGAN training.

Second, the GAN training phase has been done as prescribed by the paper. The generator is trained to generate colorized images, while the discriminator is trained to distinguish between real and fake/generated images. The process involves optimizing both networks in an adversarial manner. First the discriminator is trained, the AB channel generated by the generator is concatenated with the L channel, creating composite images. These composite images are passed through the discriminator network, and the Binary Cross Entropy with Logits Loss is calculated to determine the accuracy of its predictions. The discriminator is then trained on real images, consisting of the concatenated original AB channel and L channel, are passed through the discriminator network, and the Binary Cross Entropy with Logits Loss is calculated. The final discriminator loss is calculated as the average of the losses from both the composite and real images. This average loss reflects the discriminator's overall performance in distinguishing between real and fake/generated images. The weights of the discriminator are then updated based on this loss. (NOTE: If the discriminator is too strong a weight decay can be added to it to stablize the training process). Next, the generator is trained to generate more convincing and realistic colorized images. The generator's objective is to fool the discriminator by generating colorized images that are indistinguishable from real images. The composite image, consisting of the concatenated AB and L channels, is again passed through the discriminator. The loss calculated from it is used to evaluate the generator. Then the L1 loss is calculated between the generated AB channel and the ground truth AB channel, the L1 loss is scaled by a lambda value of 100. The generator loss is the sum of the GAN loss and the scaled L1 loss and the weights of the model are updated accordingly.

# Results
  ![](https://github.com/Moddy2024/Image-Colorization/blob/main/results/727.png)
  ![](https://github.com/Moddy2024/Image-Colorization/blob/main/results/7752.png)
  ![](https://github.com/Moddy2024/Image-Colorization/blob/main/results/824.png)
  ![](https://github.com/Moddy2024/Image-Colorization/blob/main/results/856.png)
  ![](https://github.com/Moddy2024/Image-Colorization/blob/main/results/858.png)
  ![](https://github.com/Moddy2024/Image-Colorization/blob/main/results/859.png)
  ![](https://github.com/Moddy2024/Image-Colorization/blob/main/results/howrahbus.png)
  ![](https://github.com/Moddy2024/Image-Colorization/blob/main/results/kolkatastreet.png)
  ![](https://github.com/Moddy2024/Image-Colorization/blob/main/results/chowringhee.png)
  ![](https://github.com/Moddy2024/Image-Colorization/blob/main/results/lonkolbus.png)
  ![](https://github.com/Moddy2024/Image-Colorization/blob/main/results/beatles.png)
 # Citation
```bash
@inproceedings{isola2018pix2pix,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A.},
  booktitle={CVPR},
  year={2018}
}
```
