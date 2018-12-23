# Embedding_GAN
Embedding Adversarial Learning for Vehicle Re-Identification
This code is for the work: ******

Note that: the complete code for training and testing will be opened in 2019. 

## Requirements
- pytorch
- jupyter notebook
- Linux
- Python 2 or 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation
- Install PyTorch and dependencies from http://pytorch.org
- Install Torch vision from the source.
```bash
git clone https://github.com/pytorch/vision
cd vision
python setup.py install
```
- Install python libraries [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate).
```bash
pip install visdom
pip install dominate
```
- Clone this repo:
```bash
git clone https://github.com/yanbai1993/Embedding_GAN.git
cd Embedding GAN
```

###  visualization for hard negative
Based on Embedding GAN, you can generate hard negative samples. To better illustrate the hard negatives generation procedure, we provided the test code and several models (six models under different training stages). 
The models can be downloaded from "https://pan.baidu.com/s/1vkmccegB5epCa48C7pANbQ". 
You need to put the models into 'checkpoints' dir. 

###  performance test



