# SolarDetection
 
The goal of this project is to detect solar panels in satellite images using deep learning.
Our model is based on [U-net](https://arxiv.org/abs/1505.04597) and trained on satellite image from the USA and France.
In addition to creating a model able to accurately detect solar panels we wanted to test the model's ability to generalize to other geographic areas. 

![ea6ac42f-7804-472b-a132-787fd9fc3092](https://user-images.githubusercontent.com/55833530/143678271-2a9c016a-8ab8-425d-a9f7-5f611b461d66.png)

## Installation
To configure an environment to run this project we recommend using conda.
A `requirements.txt` is included in the repository.
To set up the conda environment, execute the following lines:
 ```
conda create -n solar-detection python=3.9
conda activate solar-detection
pip install -r requirements.txt 
```

The two main modules are `data_generation.py` handling the generation and processing of the data, and `train_model.py` for the training of the model.

## Data Processing
