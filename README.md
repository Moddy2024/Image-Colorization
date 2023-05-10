# Image-Colorization


My implementation of ["Image-to-Image Translation with Conditional Adversarial Networks"](https://arxiv.org/pdf/1611.07004v3.pdf) is based on the paper by Isola et al. The encoder is a pre-trained VGG 19 network without batch normalization (BN) layers, which is used to extract the content and style features from the content image and the style image, respectively. The encoded features of the content and style image are collected and then both of these features are sent to the AdaIN layer for style transfer in the feature space.
![](https://github.com/Moddy2024/AdaIN-Style-Transfer/blob/main/results/architecture.png)
AdaIN performs style transfer in the feature space by transferring feature statistics, specifically the channel-wise mean and variance. AdaIN layer takes the features produced by the encoder of both the content image and style image and simply aligns the mean and variance of the content feature to match those of the style feature, producing the target feature ***t***.

A decoder network is then trained to generate the final stylized image by inverting the AdaIN output ***t*** back to the image space generating the stylized image. The decoder mostly mirrors the encoder, with all pooling layers replaced by nearest up-sampling to reduce checkerboard effects. Reflection padding has been used to avoid border artifacts on the generated image.
 
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

Once you have these dependencies installed, you can clone the AdaIN Style Transfer repository from GitHub:
```bash
https://github.com/Moddy2024/Image-Colorization.git
```
# Key Directories and Files
* [ADAIN.ipynb](https://github.com/Moddy2024/AdaIN-Style-Transfer/blob/main/ADAIN.ipynb) - In this Jupyter Notebook, you can find a comprehensive walkthrough of the data pipeline for AdaIN style transfer, which includes steps for downloading the dataset, preprocessing the data, details on the various data transformations and data loader creation steps,  along with visualizations of the data after transformations and moving the preprocessed data to the GPU for model training. There's also the implementation of the AdaIN style transfer, the architecture of the model used for training and the whole training process.
* [prediction.ipynb](https://github.com/Moddy2024/AdaIN-Style-Transfer/blob/main/prediction.ipynb) - This notebook demonstrates how to perform style transfer on images using the pre-trained model. As this is a Adaptive Style Transfer so any style image and content image can be used.
* [results](https://github.com/Moddy2024/AdaIN-Style-Transfer/tree/main/results) - This directory contains the results from some of the test images that have been collected after the last epoch of training.
* [test-style](https://github.com/Moddy2024/AdaIN-Style-Transfer/tree/main/test-style) - This directory contains a collection of art images sourced randomly from the internet, which are intended to be used for testing and evaluation purposes.
* [test-content](https://github.com/Moddy2024/AdaIN-Style-Transfer/tree/main/test-content) - This directory contains a collection of content images sourced randomly from the internet, which are intended to be used for testing and evaluation purposes.
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
  ![](https://github.com/Moddy2024/AdaIN-Style-Transfer/blob/main/results/victoria-memorial-womanwithhat-matisse.png)
  ![](https://github.com/Moddy2024/AdaIN-Style-Transfer/blob/main/results/lenna-picasso-seatednude.png)
  ![](https://github.com/Moddy2024/AdaIN-Style-Transfer/blob/main/results/goldengate-starrynight.jpg)
 # Citation
```bash
@inproceedings{isola2017pix2pix,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A.},
  booktitle={CVPR},
  year={2017}
}
```
