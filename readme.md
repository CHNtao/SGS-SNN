# SGS-SNN
Surrogate Gradient Scaling for Directly Training Spiking Neural Networks

This paper has been submitted to ***Applied Intelligence***. The complete code will be made public when our paper is accepted.

## Requirements
*  Python 3.9.7
*  Torch 1.10.1
*  Torchvision 0.11.2
*  Numpy 1.22.0


## Datasets
*  [CIFAR10](http://www.cs.toronto.edu/~kriz/cifar.html) 
*  [CIFAR100](http://www.cs.toronto.edu/~kriz/cifar.html)
*  [DVS-CIFAR10](https://figshare.com/s/d03a91081824536f12a8)
## TO DO
*  layers.py: surrogate gradient scaling function corresponds to Eq. (5) in our paper.
*  my_main.py: Specify the dataset location ; Specify .pth file location. [Inference model](https://drive.google.com/file/d/1g2cGOKT_xd6GdtFBVZbWSAj_0rNMyy1P/view?usp=drive_link)
*  train.py: release soon

## Usage
```python
python my_main.py

