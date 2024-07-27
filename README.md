# SelM

<div align="center">

<h2>
SelM: Selective Mechanism based Audio-Visual Segmentation
</h2>

<h4>
<b>
Jiaxu Li, Songsong Yu, Yifan Wang*, Lijun Wang, Huchuan Lu
</b>
</h4>
</div>
This repository contains code for "SelM: Selective Mechanism based Audio-Visual Segmentation" (<b>ACM MM 2024 Oral</b>).

## Overview
![Overview](images/Overview.png)



## Environment Prepare
Our Code was tested upon a conda environment. 

You can install conda by this link [Conda](https://docs.conda.io/en/latest/miniconda.html) and then create an environment as follows:

`conda create -n selm python=3.9 `

`conda activate catr`

We use Pytorch 2.0.1 with CUDA-11.7 as our default setting, install Pytorch by pip as below

`pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2`

Notice : Mamba-ssm [Link](https://github.com/state-spaces/mamba) require CUDA 11.6+ , you might have to update your CUDA.

for other required packages:

`pip install -r requirements.txt`

## Dataset and Pretrained Backbone
For AVSBench Dataset ,please refer to this link [AVSBench](https://github.com/OpenNLPLab/AVSBench) to download the datasets

For Pretrained Backbone(ResNet50、PVT-V2、VGGish),please refer to this [link](https://drive.google.com/drive/folders/1386rcFHJ1QEQQMF6bV1rXJTzy8v26RTV?usp=sharing) to download.

You can placed the dataset and pretrained backbone to the directory `data` `pretrained backbone`
Notice : Don't forget to change the paths of data and model in `config.py`

## Pretrained Model

## Train
## Test
## Ciation

