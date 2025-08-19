## Generative-Label-Fused-Network
GARE-Net: Geometric Contextual Aggregation and
Regional Contextual Enhancement Network for
Image-Text Matching

## Introduction

This is the source code of Generative Label Fused Network, an approch for Image-Text Matching. It is built on top of the SCAN (by [Kuang-Huei Lee](https://github.com/kuanghuei/SCAN)) and PFAN (by  [Yaxiong Wang]( [HaoYang0123/Position-Focused-Attention-Network: Position Focused Attention Network for Image-Text Matching (github.com)](https://github.com/HaoYang0123/Position-Focused-Attention-Network) ))in PyTorch.

## Requirements and Installation
We recommended the following dependencies.

* Python 3.9.18
* [PyTorch](http://pytorch.org/) 2.2.2
* [NumPy](http://www.numpy.org/) (>1.26.2)
* [TensorBoard](https://github.com/TeamHG-Memex/tensorboard_logger)

The workflow of Gare-Net

<img src="https://raw.githubusercontent.com/smileslabsh/Generative-Label-Fused-Network/main/figures/main.png" width="745" alt="workflow" /> 

## Download data
Download the dataset files. We use the dataset files created by SCAN [Kuang-Huei Lee](https://github.com/kuanghuei/SCAN). The position information of images can be downloaded from [here](https://github.com/HaoYang0123/Position-Focused-Attention-Network/tree/master) 

## Training new models

To train Flickr30K and MS-COCO models:
```bash
sh train.sh
```
## Results

