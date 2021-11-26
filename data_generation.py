from numpy.lib.type_check import imag
import rasterio
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image, ImageDraw
from tqdm import tqdm
import os
import shutil
import glob
import utm


class DatasetGenerator:
    """Generate datasets of solar panel image-mask pairs.
    

    
    Args:
        dataset_path (str): path to the directory where the dataset will be generated. 
            If the path does not exist the the directory will be created, including intermediary directories in the path.
        
        sample_size (int): Width and height in pixels of generated image samples.
            Each sample contains the centroid of a solar panel polygon.
        
        border_ratio (float): Size of border, as fraction of sample_size
            To avoid the center pixels in the image sample always containing a solar panel,
            the image samples are sampled such that the solar panel centroids 
            are uniformly ditributed within a square in the center of the image samples.
            The border_ratio controlls the size of the border of the image samples 
            where centroids can not be contained.
            If set to 0, solar panel centroid can be contained anywhere in the image samples.
            If set to 1, solar panel centroid will be in the center of the image samples.
    """
    def __init__(self, dataset_path, sample_size=128, border_ratio=.25, shuffle=True, max_num_samples=None, clear_data=True, test_split=None):
        self.set_dataset_path(dataset_path)
        self.sample_counter = 0
        self.sample_size= sample_size
        self.border_ratio = border_ratio
        self.shuffle = shuffle
        self.clear_data = clear_data
        self.test_split = test_split
        self.max_num_samples = max_num_samples
    
    def set_dataset_path(self, dataset_path):
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)
        image_path = os.path.join(dataset_path, "images")
        mask_path = os.path.join(dataset_path, "masks")
        if not os.path.exists(image_path):
            os.makedirs(image_path)
        if not os.path.exists(mask_path):
            os.makedirs(mask_path)
        
        self.dataset_path = dataset_path
        self.image_path = image_path
        self.mask_path = mask_path

    def _clear_data(self):
        """Clears all files in the dataset directory."""
        for directory in ["images", "masks"]:
            files = glob.glob(os.path.join(self.dataset_path, directory, "*"))
            for f in files:
                os.remove(f)
        
        delete_directories = ["images_old", "masks_old", "train", "test"]
        for directory in delete_directories:
            if os.path.exists(os.path.join(self.dataset_path, directory)):
                shutil.rmtree(os.path.join(self.dataset_path, directory))
        self.sample_counter = 0
    
    def read_polygon_file(self, polygon_file):
        """Imports a geojson file containing the polygons of the solar panels as a dictionary."""
        with open(polygon_file, mode='r') as f:
            polygon_data = json.load(f)
        self.polygon_data = polygon_data["features"]
    
    def get_image_metadata(self, image_dir, file_format=".tif"):
        """Returns a set of distinct image file names mentioned in the polygon file.
        
        The elements in the returned set are file path - file name pairs.
        The path is used later used to import the images, while the file name is later 
        used to iterate over the images mentionned in the polygon file one by one.
        """
        image_metadata = []
        for panel in self.polygon_data:
            city_dir = os.path.join(image_dir, panel["properties"]["city"])
            file_name = panel["properties"]["image_name"]
            metadata = os.path.join(city_dir, file_name + file_format), file_name
            image_metadata.append(metadata)
        
        # self.image_file_names = set(image_files)
        return set(image_metadata)
    
    def get_polygons_in_image(self, image_file):
        """Returns a list of polygons and corresponding centroids contained in the given image."""
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
        """Returns a binary mask indicating solar panels in the image."""
        img_w = image.shape[1]
        img_h = image.shape[0]
        mask = Image.new('L', (img_w, img_h), 0)
        for poly in polygons:
            poly_int_coords = self.poly_coords_to_int(poly)
            ImageDraw.Draw(mask).polygon(poly_int_coords, outline=255, fill=255)
        mask = np.array(mask, dtype='uint8')
        return mask

    def get_sample_bounds(self, image, centroid):
        """Retruns the bounds of a image sample.
        
        The image samples are sampled such that the solar panel centroids 
        are uniformly ditributed within a square in the center of the image samples.
        The padding_ratio controlls the size of the border of the image samples 
        where centroids can not be contained.
        """
        img_w = image.shape[1]
        img_h = image.shape[0]
        xcentr = round(centroid[0])
        ycentr = round(centroid[1])
        border = round(self.sample_size*self.border_ratio)
        sample_xmin_L = max(xcentr - self.sample_size + border, 0)
        sample_xmin_U = min(xcentr - border, img_w - self.sample_size)
        
        sample_ymin_L = max(ycentr - self.sample_size + border, 0)
        sample_ymin_U = min(ycentr - border, img_h - self.sample_size)
        
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

    def generate_samples(self, polygon_file, image_dir):
        if self.clear_data:
            self._clear_data()

        print("Generating dataset...")
        self.polygon_file = polygon_file
        self.image_dir = image_dir
        self.read_polygon_file(polygon_file)
        num_panels = len(self.polygon_data)
        num_samples = min(num_panels, self.max_num_samples) if self.max_num_samples is not None else num_panels
        # Looping through image files first so that each image file will be opened and closed only once
        image_metadata = self.get_image_metadata(image_dir)
        with tqdm(total=num_samples) as progress_bar:
            for image_path, image_name in image_metadata:
                image = self.import_image(image_path)
                polygons, centroids = self.get_polygons_in_image(image_name)
                mask = self.get_mask(image, polygons)
                for centroid in centroids:
                    sample_xmin, sample_xmax, sample_ymin, sample_ymax = self.get_sample_bounds(image, centroid)
                    image_sample = image[sample_ymin:sample_ymax, sample_xmin:sample_xmax]
                    mask_sample = mask[sample_ymin:sample_ymax, sample_xmin:sample_xmax]
                    Image.fromarray(image_sample).save(os.path.join(self.dataset_path, "images", f"i_{self.sample_counter}.png"))
                    Image.fromarray(mask_sample).save(os.path.join(self.dataset_path, "masks", f"m_{self.sample_counter}.png"))
                    self.sample_counter += 1
                    progress_bar.update(1)
                    if self.sample_counter == num_samples:
                        break
                if self.sample_counter == num_samples:
                    break
        print("Dataset generation complete.")
        if self.shuffle:
            self.shuffle_dataset()
        if self.test_split is not None:
            self.split_dataset(self.test_split)
    
    def shuffle_dataset(self):
        print("Shuffling dataset...")
        os.rename(self.image_path, os.path.join(self.dataset_path, "images_old"))
        os.rename(self.mask_path, os.path.join(self.dataset_path, "masks_old"))
        os.makedirs(self.image_path)
        os.makedirs(self.mask_path)

        sample_index_shuffled = np.random.permutation(self.sample_counter)
        for old_index, new_index in enumerate(tqdm(sample_index_shuffled)):
            old_image_path = os.path.join(self.dataset_path, "images_old", f"i_{old_index}.png")
            new_image_path = os.path.join(self.image_path, f"i_{new_index}.png")
            os.rename(old_image_path, new_image_path)

            old_mask_path = os.path.join(self.dataset_path, "masks_old", f"m_{old_index}.png")
            new_mask_path = os.path.join(self.mask_path, f"m_{new_index}.png")
            os.rename(old_mask_path, new_mask_path)
        
        os.rmdir(os.path.join(self.dataset_path, "images_old"))
        os.rmdir(os.path.join(self.dataset_path, "masks_old"))
        print("Dataset shuffling complete.")
    
    def split_dataset(self, test_split):
        train_path = os.path.join(self.dataset_path, "train")
        test_path = os.path.join(self.dataset_path, "test")
        print(train_path)
        if not os.path.exists(train_path):
            os.makedirs(os.path.join(train_path, "images"))
            os.makedirs(os.path.join(train_path, "masks"))
        if not os.path.exists(test_path):
            os.makedirs(os.path.join(test_path, "images"))
            os.makedirs(os.path.join(test_path, "masks"))
        
        sample_index_shuffled = np.random.permutation(self.sample_counter)
        self.num_test_samples = int(test_split * self.sample_counter)
        self.num_train_samples = self.sample_counter - self.num_test_samples
        for new_index in range(self.num_train_samples):
            old_index = sample_index_shuffled[new_index]
            os.rename(os.path.join(self.image_path, f"i_{old_index}.png"), os.path.join(train_path, "images", f"i_{new_index}.png"))
            os.rename(os.path.join(self.mask_path, f"m_{old_index}.png"), os.path.join(train_path, "masks", f"m_{new_index}.png"))
        for new_index in range(self.num_train_samples, self.sample_counter):
            old_index = sample_index_shuffled[new_index]
            os.rename(os.path.join(self.image_path, f"i_{old_index}.png"), os.path.join(test_path, "images", f"i_{new_index}.png"))
            os.rename(os.path.join(self.mask_path, f"m_{old_index}.png"), os.path.join(test_path, "masks", f"m_{new_index}.png"))
        
        os.rmdir(self.image_path)
        os.rmdir(self.mask_path)

    @staticmethod
    def import_image(image_file):
        with rasterio.open(image_file) as f:
            image = np.array(f.read())
            image = image.transpose((1,2,0))
            if image.shape[-1] > 3:
                print(image_file, image.shape)
        return image[:,:,:3]

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


class SegmentationDataGenerator(Sequence):
    """Generate batches of image-mask pairs.
    
    Args:
        dataset_path (str): Path to the dataset.
        image_size (int or tuple of ints): Size, in pixels, of the image samples.
            - int: equal height and width of the image samples.
            - tuple of ints: (height, width) of the image samples.
        mask_size (int or tuple of ints): Size, in pixels, of the mask samples.
            - int: equal height and width of the mask samples.
            - tuple of ints: (height, width) of the mask samples.
            Default: same as image_size
        batch_size (int): Size of the generated batches.
        shuffle (bool): Shuffle the dataset at the end of an epoch.
        rescale (float): Rescaling factor for the image and mask samples.
            Images are normally encoded with unsigned intergers between 0 and 255. 
            Scaling the values to be between 0 and 1 usually leads to better performance.
        vflip (bool): Randomly flip images vertically.
        hflip (bool): Randomly flip images horizontally.
    """
    def __init__(self, 
                 dataset_path, 
                 image_size=(128, 128), 
                 mask_size=None, 
                 batch_size=32, 
                 shuffle=True, 
                 rescale=1/255., 
                 vflip=True,
                 hflip=True): 
        self.dataset_path = dataset_path
        self.image_path = os.path.join(dataset_path, "images")
        self.mask_path = os.path.join(dataset_path, "masks")
        self.image_names = [name for name in os.listdir(self.image_path)]
        self.mask_names = [name for name in os.listdir(self.mask_path)]
        self.n_files = len(self.image_names)
        print(f"Found {self.n_files} files.")
        self.shuffle = shuffle
        if shuffle:
            self.shuffle_samples()
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.image_size = image_size
        if mask_size is None:
            mask_size = image_size
        self.mask_size = mask_size
        self.batch_size = batch_size
        self.rescale = rescale
        self.vflip = vflip
        self.hflip = hflip
        self.last_used_index = 0

    def __len__(self):
        """Returns the number of batches in an epoch."""
        return self.n_files // self.batch_size

    def __getitem__(self, index):
        """Returns a tuple (X, y) of images and mask of one batch."""
        X = np.zeros((self.batch_size, *self.image_size, 3), dtype="float32")
        y = np.zeros((self.batch_size, *self.image_size, 1), dtype="float32")

        batch_image_names = self.image_names[index*self.batch_size:(index+1)*self.batch_size]
        batch_mask_names = self.mask_names[index*self.batch_size:(index+1)*self.batch_size]
        for i, (image_name, mask_name) in enumerate(zip(batch_image_names, batch_mask_names)):
            vi = np.random.randint(2)*2-1 if self.vflip else 1
            hi = np.random.randint(2)*2-1 if self.hflip else 1
            with Image.open(os.path.join(self.image_path, image_name)) as img:
                X[i] = np.array(img, dtype="float32")[::vi,::hi,:3] * self.rescale 
                
            with Image.open(os.path.join(self.mask_path, mask_name)) as mask:
                y[i,:,:,0] = np.array(mask, dtype="float32")[::vi,::hi] * self.rescale 
        self.last_used_index = index
        return X, y
    
    def on_epoch_end(self):
        """When an epoch ends the samples are shuffled."""
        if self.shuffle:
            self.shuffle_samples()

    def next(self):
        """Returns the next batch in the epoch. Used to quickly get a batch to visualise results."""
        if self.last_used_index == self.__len__():
            return self.__getitem__(0)
        else:
            return self.__getitem__(self.last_used_index + 1)

    def shuffle_samples(self):
        """Shuffles the order of the images-masks pairs."""
        image_mask_pairs = list(zip(self.image_names, self.mask_names))
        np.random.shuffle(image_mask_pairs)
        image_names, mask_names = zip(*image_mask_pairs)
        self.image_names = image_names
        self.mask_names = mask_names
    
    @staticmethod
    def get_sample_index(name):
        return name.split("_")[1].split(".")[0]
    


if __name__=='__main__':
    polygon_path = r"..\Projet_INSA_France\DeepSolar\DATA_DeepSolar\metadata\SolarArrayPolygons.geojson"
    image_path = r"..\Projet_INSA_France\DeepSolar\DATA_DeepSolar"
    dataset_path = r"test_dataset2"

    dataset_gen = DatasetGenerator(dataset_path, max_num_samples=1000)
    # dataset_gen.clear_data()
    dataset_gen.generate_samples(polygon_path, image_path)