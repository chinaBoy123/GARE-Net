## GARE-Net: Geometric Contextual Aggregation and Regional Contextual Enhancement Network for Image-Text Matching
GARE-Net: Geometric Contextual Aggregation and
Regional Contextual Enhancement Network for
Image-Text Matching

## Introduction
This is the source code of GARE-Net. It is built on top of the CHAN in PyTorch.

<img src="https://raw.githubusercontent.com/chinaBoy123/GARE-Net/main/figures/garenet.png" width="745" alt="workflow" />

## Requirements and Installation
We recommended the following dependencies.
* Python 3.9.18
* [PyTorch](http://pytorch.org/) 2.2.2
* [NumPy](http://www.numpy.org/) (>=1.26.2)
* [TensorBoard](https://github.com/TeamHG-Memex/tensorboard_logger)

## Download data
Download the dataset files. We use the dataset files created by SCAN [Kuang-Huei Lee](https://github.com/kuanghuei/SCAN). The box features of Flickr30K and MSCOCO are extracted from the raw Flickr30K images using the bottom-up attention model from [here](https://github.com/peteanderson80/bottom-up-attention). 

## Training new models
To train Flickr30K and MS-COCO models:
```bash
sh scripts/train.sh
```

## Evaluation
```bash
sh scripts/eval.sh
```

## Results
#### Results on COCO 1K Test Split

| |Visual Backbone|Text Backbone|R1|R5|R10|R1|R5|R10|
|---|:---:|:---:|---|---|---|---|---|---|
|GARE-Net | BUTD region |GRU-base|**84.2**|**96.7**|**98.4**|**65.5**|**91.6**|**96.0**|
|GARE-Net | BUTD region |BERT-base|**84.5**|**97.0**|**99.1**|**67.4**|**92.0**|**96.6**|

#### Results on Flickr30K Test Split

| |Visual Backbone|Text Backbone|R1|R5|R10|R1|R5|R10|
|---|:---:|:---:|---|---|---|---|---|---|
|GARE-Net | BUTD region |GRU-base|**80.8**|**96.0**|**98.2**|**60.9**|**86.1**|**91.2**|
|GARE-Net | BUTD region |BERT-base|**81.9**|**95.9**|**98.6**|**65.3**|**89.4**|**93.7**|
