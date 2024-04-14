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
```
M reg   //Image samples after super-resolution
dataset   //Classification model training
├── val   //Validation set, no labels
├── test   //Test set, two folders, same as train
└── train  //Train set
    ├── field       
    └── nonfield
weights   //SAM pre-training weights and classification model weights
```

Click the links below to download the checkpoint for the corresponding model type. For testing time considerations, we recommend that you use `vit_b` with the smallest number of parameters for testing.

- `default` or `vit_h`:[ViT-H SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
- `vit_l`:[ViT-L SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
- `vit_b`:[ViT-B SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
- 
## Getting Started
The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`.
```
pip install git+https://github.com/facebookresearch/segment-anything.git
```
```
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```
