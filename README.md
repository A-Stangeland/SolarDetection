# SolarDetection

## Overview
The goal of this project is to detect solar panels in satellite images using deep learning.
Our model is based on [U-net](https://arxiv.org/abs/1505.04597) and trained on satellite image from the USA and France.
In addition to creating a model able to accurately detect solar panels we also wanted to test the model's ability to generalize to other geographic areas. 
The model performs semantic segmentation, meaning that it predicts the presence of a solar panel for every pixel in the input image. 


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

## Usage
### Generating the dataset 
The data generator class to use depends on the data source, as US and French satellite data have different sizes, and thus require different approaches. 
The US dataset is composed of images that are 5000 by 5000 pixels in sixe and usage of the ```DatasetGenerator``` class is created with images of this size in mind.
The French satellite images we had access to were 25 000 by 25000 pixels, and too large to load the whole image in memory. 
For this reason, a second data generation class, ```DatasetGeneratorGERS```, was created by adapting the data generator class for the US images. 

To generate a dataset of image samples and their corresponding binary mask ('0' if there is no panel, '1' if there is) the following arguments can be specified: 

* ```image_path```: Path to the satellite images (str)
* ```json_path```: Path to JSON file containing the panel polygons (str)
* ```dataset_path```: Path to where the dataset will be created (str)
* ```gers```: Shuffle after generating samples (Bool)
* ```image_size```: Generated image sample size, samples will be ```image_size``` by ```image_size``` pixels in size (int)
* ```shuffle```: Shuffle after generating samples (Bool)
* ```test_split```: Ratio of samples in the test set (Float)

Execute the below command to generate the dataset: 
```
python data_generation.py --dataset_path dataset_path --json_path json_path --image_size image_size --shuffle shuffle --test_split test_split
```

Using the default values chosen by us the above command becomes: 
```
python data_generation.py --dataset_path "./data/images" --json_path "./data" --image_size 128 --shuffle True --test_split 0.25
```

### Data Augmentation
Data augmentation is used in the project to get the most out of the accessible dataset. The methods used are vertical and horizontal flips of image samples and its corresponding binary mask. 
The image sample has a 50% likelihood of being flipped along each axis, essentially making it possible to generate four total training samples from each training sample in the original dataset. The ground truth label is flipped along with the sample. 

## Data Processing

## Polygon JSON schema
The dataset generation is adapted for a dataset of satellite imagery with a corresponding JSON file containing the solar panel polygons, and this JSON file has a pre-defined structure that should be followed to generate new samples and ground truth labels without needing to modify the ```data_generation.py``` script.  
The structure of the JSON file is given below: 

```python
{
    "geometry": {
        "coordinates": [
            [-119.83998,36.92599116],
            ...
        ]
    },
    "properties": {
        "polygon_id": 1,
        "image_name": "11ska460890",
        "datum": "NAD83",
        "projection_zone": "11",
        "resolution": 0.3,
        "city": "Fresno",
        "nw_corner_of_image_latitude": 36.92633611,
        "nw_corner_of_image_longitude": -119.8516222,
        "se_corner_of_image_latitude": 36.91323333,
        "se_corner_of_image_longitude": -119.8343,
        "centroid_latitude_pixels": 107.6184581,
        "centroid_longitude_pixels": 3286.151487,
        "polygon_vertices_pixels": [
            [3360.495069,131.6311637],
            ...
        ]
    }
}
```

## Further Development
### Data Augmentation
Adding more transformation than randomly flipping the image samples along its horizontal and vertical axes is likely to make the model more robust. 
We would recommend implementing rotations and zooms as a next step in improving data augmentation but would refrain from using transformations like shears and elastic deformations to preserve the characteristic shape of solar panels in the training set. 
