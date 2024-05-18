# Detectron2-Custom-Object-Detection-And-Segmentation

<img src="https://github.com/facebookresearch/detectron2/blob/main/.github/Detectron2-Logo-Horz.svg" width="300">

Detectron2 to For custom objects : Detectron2 is a popular PyTorch based modular computer vision model library.
It is the second iteration of [Detectron](https://github.com/facebookresearch/Detectron/), originally written in Caffe2.
The Detectron2 system allows you to plug in custom state of the art computer vision technologies into your workflow.
It supports a number of computer vision research projects and production applications in various AI tech industries.

## Model Zoo and Baselines
It provides a large set of baseline results and trained models available for download in the [Detectron2 Model Zoo](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md).

This repository allows you to try on your own vision applications by replacing the pretarined models mentioned above according to your usecase.

## Training and inferencing
##### 1. Data Preparation
First gather the data from the web browser by running the script 'image_downloader.py' with mentioning the query string that refers what kind of data you want to install for your usecase.
```
query_string = "Various type of Vehicles with clear Number plates"

python image_downloader.py
```


### 2. Labelling the data
After gathering data, we have to label it which our model should detect and segment. Here I have used a software called labelme to label the instances and get labelled json data file with respect to each image.
Incase if you don't know how to use this software, just go through this page [Labelme](https://datagen.tech/guides/image-annotation/labelme/) you will get how to do.
![](Labelme)
