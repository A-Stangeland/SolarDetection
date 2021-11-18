# from numpy.lib.type_check import imag
import rasterio
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image, ImageDraw
import os
import glob
import utm


class DatasetGenerator:
    def __init__(self, dataset_dir, sample_size=128) -> None:
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        self.dataset_dir = dataset_dir
        
        self.sample_counter = 0
        self.sample_size= sample_size
    
    def clear_data(self):
        for dir1 in ["image", "label"]:
            files = glob.glob(os.path.join(self.dataset_dir, dir1, "*"))
            for f in files:
                os.remove(f)
        for dir1 in ["train", "test"]:
            for dir2 in ["image", "label"]:
                files = glob.glob(os.path.join(self.dataset_dir, dir1, dir2, "*"))
                for f in files:
                    os.remove(f)
        self.sample_counter = 0
    
    def read_polygon_file(self, polygon_file):
        with open(polygon_file, mode='r') as f:
            polygon_data = json.load(f)
        
        self.polygon_data = polygon_data["features"]
        return polygon_data["features"]
    
    def get_image_metadata(self, image_dir, file_format=".tif"):
        image_metadata = []
        for panel in self.polygon_data:
            city_dir = os.path.join(image_dir, panel["properties"]["city"])
            file_name = panel["properties"]["image_name"]
            metadata = os.path.join(city_dir, file_name + file_format), file_name
            image_metadata.append(metadata)
        
        # self.image_file_names = set(image_files)
        return set(image_metadata)
    
    def get_polygons_in_image(self, image_file):
        polygons = []
        centroids = []
        for panel in self.polygon_data:
            # Check if the panel is inclueded in the currently used image
            if panel["properties"]["image_name"] != image_file:
                continue
            if panel["properties"]["centroid_longitude_pixels"] == "_NaN_" \
            or panel["properties"]["centroid_latitude_pixels"] == "_NaN_":
                continue
            
            polygons.append(panel["properties"]["polygon_vertices_pixels"])
            xcentr = panel["properties"]["centroid_longitude_pixels"]
            ycentr = panel["properties"]["centroid_latitude_pixels"]
            centroids.append([xcentr, ycentr])
        return polygons, centroids
    
    def get_mask(self, image, polygons):
        img_w = image.shape[1]
        img_h = image.shape[0]
        mask = Image.new('L', (img_w, img_h), 0)
        for poly in polygons:
            poly_int_coords = self.poly_coords_to_int(poly)
            ImageDraw.Draw(mask).polygon(poly_int_coords, outline=255, fill=255)
        mask = np.array(mask, dtype='uint8')
        return mask

    def get_sample_bounds(self, image, centroid, padding_ratio=.25):
        img_w = image.shape[1]
        img_h = image.shape[0]
        xcentr = round(centroid[0])
        ycentr = round(centroid[1])
        padding = round(self.sample_size*padding_ratio)
        sample_xmin_L = max(xcentr - self.sample_size + padding, 0)
        sample_xmin_U = min(xcentr - padding, img_w - self.sample_size)
        
        sample_ymin_L = max(ycentr - self.sample_size + padding, 0)
        sample_ymin_U = min(ycentr - padding, img_h - self.sample_size)
        
        if sample_xmin_L < sample_xmin_U:
            sample_xmin = np.random.randint(sample_xmin_L, sample_xmin_U)
        else:
            if sample_xmin_L == 0:
                sample_xmin = 0
            else:
                sample_xmin = img_w - self.sample_size
        
        if sample_ymin_L < sample_ymin_U:
            sample_ymin = np.random.randint(sample_ymin_L, sample_ymin_U)
        else:
            if sample_ymin_L == 0:
                sample_ymin = 0
            else:
                sample_ymin = img_h - self.sample_size
        
        sample_xmax = sample_xmin + self.sample_size
        sample_ymax = sample_ymin + self.sample_size

        return sample_xmin, sample_xmax, sample_ymin, sample_ymax

    def generate_samples(self, polygon_file, image_dir, super_sample=False):
        self.polygon_file = polygon_file
        self.image_dir = image_dir
        
        polygon_data = self.read_polygon_file(polygon_file)
        
        # Looping through image files first so the files can be opened and closed seperately
        image_metadata = self.get_image_metadata(image_dir)
        for image_path, image_name in image_metadata:
            print(image_path, image_name)
            image = self.import_image(image_path)
            polygons, centroids = self.get_polygons_in_image(image_name)
            mask = self.get_mask(image, polygons)
            for i in range(len(polygons)):
                sample_xmin, sample_xmax, sample_ymin, sample_ymax = self.get_sample_bounds(image, centroids[i])
                image_sample = image[sample_ymin:sample_ymax, sample_xmin:sample_xmax]
                mask_sample = mask[sample_ymin:sample_ymax, sample_xmin:sample_xmax]
                print(self.dataset_dir + "image/" f"i_{self.sample_counter}.png")
                Image.fromarray(image_sample).save(self.dataset_dir + "image/" f"i_{self.sample_counter}.png")
                Image.fromarray(mask_sample).save(self.dataset_dir + "label/" f"m_{self.sample_counter}.png")
                self.sample_counter += 1
    
    def split_dataset(self, test_split=.25):
        sample_index_shuffled = np.random.permutation(self.sample_counter)
        self.num_test_samples = int(test_split * self.sample_counter)
        self.num_train_samples = self.sample_counter - self.num_test_samples

        for i in range(self.num_train_samples):
            sample_index = sample_index_shuffled[i]
            os.rename(self.dataset_dir + f"image/i_{sample_index}.png", self.dataset_dir + f"train/image/i_{i}.png")
            os.rename(self.dataset_dir + f"label/m_{sample_index}.png", self.dataset_dir + f"train/label/m_{i}.png")

        for i in range(self.num_train_samples, self.sample_counter):
            sample_index = sample_index_shuffled[i]
            os.rename(self.dataset_dir + f"image/i_{sample_index}.png", self.dataset_dir + f"test/image/i_{i}.png")
            os.rename(self.dataset_dir + f"label/m_{sample_index}.png", self.dataset_dir + f"test/label/m_{i}.png")
            
    @staticmethod
    def add_folder_slash(folder):
        if folder[-1] not in ["/", "\\"]:
            folder += "/"
        return folder
    
    @staticmethod
    def import_image(image_file):
        with rasterio.open(image_file) as f:
            image = np.array(f.read())
            image = image.transpose((1,2,0))
        return image

    @staticmethod
    def poly_coords_to_int(polygon):
        """Rounds the coordinates of the polygon to nearest integer."""
        poly_int_coords = list(map(lambda vertex: (int(vertex[0]), int(vertex[1])), polygon))
        return poly_int_coords

    @staticmethod
    def get_bbox(polygon):
        """Returns the bounding box of the polygon."""
        xmin = polygon[0][0]
        xmax = polygon[0][0]
        ymin = polygon[0][1]
        ymax = polygon[0][1]
        for x, y in polygon[1:]:
            xmin = min(xmin, x)
            xmax = max(xmax, x)
            ymin = min(ymin, y)
            ymax = max(ymax, y)
        return xmin, xmax, ymin, ymax


class ImageStream:
    def __init__(self, data_dir, batch_size=32, sample_size=128, rescale=1/255.) -> None:
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.sample_counter = 0
        self.batch_counter = 0
        self.rescale = rescale
    
    def next(self):
        image_batch = np.zeros((self.batch_size, self.sample_size, self.sample_size, 3), dtype="float32")
        mask_batch = np.zeros((self.batch_size, self.sample_size, self.sample_size, 1), dtype="float32")
        for i in range(self.batch_size):
            with Image.open(os.path.join(self.data_dir, "image", f"i_{self.sample_counter}.png")) as img:
                image_batch[i] = np.array(img)
                
            with Image.open(os.path.join(self.data_dir, "label", f"m_{self.sample_counter}.png")) as mask:
                mask_batch[i] = np.array(mask).reshape((self.sample_size, self.sample_size, 1))
            
            self.sample_counter += 1
        
        image_batch *= self.rescale
        mask_batch *= self.rescale
        
        self.batch_counter += 1
        return image_batch, mask_batch
    
    def load_dataset(self, num_samples):
        image_data = np.zeros((num_samples, self.sample_size, self.sample_size, 3), dtype="float32")
        mask_data = np.zeros((num_samples, self.sample_size, self.sample_size, 1), dtype="float32")
        
        for i in range(num_samples):
            print(i)
            with Image.open(os.path.join(self.data_dir, "image", f"i_{i}.png")) as img:
                image_data[i] = np.array(img)[:,:,:3]
                
            with Image.open(os.path.join(self.data_dir, "label", f"m_{i}.png")) as mask:
                mask_data[i] = np.array(mask).reshape((self.sample_size, self.sample_size, 1))
        
        image_data *= self.rescale
        mask_data *= self.rescale
        
        return image_data, mask_data

class SegmentationDataGenerator(Sequence):
    def __init__(self, image_dir, mask_dir, target_size=(128,128), batch_size=32, shuffle=False, data_aug_args=None):
        # super().__init__()
        if data_aug_args is None:
            data_aug_args = dict(rescale=1/255.)
        self.data_aug_args = data_aug_args

        self.image_datagen = ImageDataGenerator(**data_aug_args)
        self.mask_datagen = ImageDataGenerator(**data_aug_args)

        self.image_generator = self.image_datagen.flow_from_directory(
            image_dir,
            color_mode="rgb",
            class_mode=None,
            target_size=(128, 128),
            batch_size=batch_size,
            shuffle=shuffle,
            classes=["image"])
            
        self.mask_generator = self.mask_datagen.flow_from_directory(
            mask_dir,
            color_mode="grayscale",
            class_mode=None,
            target_size=(128, 128),
            batch_size=batch_size,
            shuffle=shuffle,
            classes=["label"])


        self.__len__ = self.image_generator.__len__

    # def __len__(self):
        # self.image_generator.__len__()

    def __getitem__(self, index):
        return self.image_generator.__getitem__(index), self.mask_generator.__getitem__(index)


if __name__=='__main__':
    polygon_file = r"Projet_INSA_France\DeepSolar\DATA_DeepSolar\metadata\SolarArrayPolygons.geojson"
    image_dir = r"Projet_INSA_France\DeepSolar\DATA_DeepSolar"
    dataset_dir = r"test_dataset"

    dataset_gen = DatasetGenerator(dataset_dir)
    dataset_gen.clear_data()
    dataset_gen.generate_samples(polygon_file, image_dir)
    dataset_gen.split_dataset()