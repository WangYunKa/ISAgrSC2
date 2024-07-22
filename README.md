# LiSegAgr
## Labeled Instance Segmentation for Agricultural Remote Sensing Images through Iterative SAM
![image](https://github.com/user-attachments/assets/bc0b6fd6-90a1-4842-97b0-0ac48086cc35)


The precise segmentation of agricultural remote sensing images is pivotal for the effective monitoring and management of cultivated land resources.

We introduce AgriSeIS and LiSegAgr, an unsupervised segmentation framework for agricultural remote sensing images based on iterative use of SAM model. This innovative approach methodology has achieved state-of-the-art performance in terms of completeness and accuracy for instance segmentation of agricultural remote sensing images, with the IoU value for segmented area reaching as high as 0.917, tested on NWPU-RESISC45, DeepGlobe and USGS datasets.

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
[Real-ESRGAN.](https://github.com/xinntao/Real-ESRGAN/tree/master)

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

## Getting Started
The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. First you need to download the Segment Anything Model.
```
pip install git+https://github.com/facebookresearch/segment-anything.git
```
In order to run SAM, you also need to install the following dependencies.
```
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```
### Segmentation
The M reg folder contains sample images that can be used to run the segmentation program. You can add your pictures in it and modify the value of `i` in `Iterative_segmentation.py` and `Refined_segmentation.py` (pictures can be of any size, and the program will automatically resize the running photo).

The following command performs preliminary iterative segmentation:
```
python Iterative_segmentation.py
```
If you need Refined segmentation, you need to run the following command afterwards:
```
python Refined_segmentation.py
```
If you need to modify the SAM pre-training weights used, just uncomment the following code accordingly.
```
# sam_checkpoint = "weights/sam_vit_h_4b8939.pth"
# sam_checkpoint = "weights/sam_vit_b_01ec64.pth"
sam_checkpoint = "weights/sam_vit_l_0b3195.pth"
# model_type = "vit_b"
model_type = "vit_l"
# model_type = "default"
```
The results of the program will be saved in the Run folder.
### Classification
Before training the classification model, you can prepare the training data yourself and organize your data in the form of the `dataset`.
You can also use the segmentation results of the segmentation model to generate your own training set, using the following steps:

Run the `json_file_generation()` function in `Plot_cropping.py` to get the json file of a certain picture, and use `X-anylabeling` or other labeling software to label it while the cultivated land is marked as `1`, the non-cultivated land is marked as `0`.

Then, save your labeled json file in the `Label` folder with corresponding images, and run `Acquisition_of_cultivated_land()` and `Acquisition_of_non_cultivated_land()` function, thereby obtaining the intercepted plot for training.

Then，training!
```
python Classification_training.py
```
If you want to evaluate your model:
```
python Evaluate_classification.py
```

### Further experimental results demonstrate the effectiveness of our method.
![image](https://github.com/user-attachments/assets/8ede2024-7de8-43eb-b76e-4f15c2bd997c)
![image](https://github.com/user-attachments/assets/1955cb34-0544-4552-b0f2-6d31e39fe869)



## References
- [Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data](https://github.com/xinntao/Real-ESRGAN/tree/master)
- [Segment Anything](https://github.com/facebookresearch/segment-anything)
