# ISAgrSC2
## Agricultural Remote Sensing Image Segmentation and Cultivated Land Classification
![image](https://github.com/WangYunKa/ISAgrSC2/assets/113222930/5a49894a-cda5-4fc9-8ac4-548210422e8d)

The precise segmentation of agricultural remote sensing images is pivotal for the effective monitoring and management of cultivated land resources.

We introduce ISAgrSC2, an unsupervised segmentation framework for agricultural remote sensing images based on iterative use of SAM model. This innovative approach methodology has achieved state-of-the-art performance in terms of completeness and accuracy for instance segmentation of agricultural remote sensing images, with the IoU value for segmented area reaching as high as 0.917, tested on NWPU-RESISC45, DeepGlobe and USGS datasets.

## Pipeline
![image](https://github.com/WangYunKa/ISAgrSC2/assets/113222930/41be6ddb-beeb-4ce5-af2f-d9b44c3d2720)

The pipeline of our method. Through iterative application of SAM segmentation and refined segmentation on incompletely segmented regions, we obtain segmentation results for remote sensing images. We then enhance the ResNet50 network through the incorporation of attention mechanisms for classification. By integrating two models, we achieved precise instance segmentation results for cultivated land.

## Install
```
git clone https://github.com/WangYunKa/ISAgrSC2.git
pip install -r requirements.txt
```

## Super resolution
When you need to use your own pictures for testing, please use Real-ESRGAN super-resolution technology to perform four times super-resolution on your own pictures.
[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN/tree/master)

## Data and Pretrained Models

dataset
├── val //5张遥感图片，有标签
├── test   //3张遥感图片，无标签，在这个任务中没有用到
└── train  //为空，通过`python preprocess.py`随机采样生成
    ├── images       
    └── labels
FP16-ViT-B-32.pt
FP16-ViT-B-16.pt
FP16-ViT-L-14.pt
FP16-ViT-L-14-336px.pt

## Getting Started

```pip install git+https://github.com/facebookresearch/segment-anything.git```
