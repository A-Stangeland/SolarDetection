import argparse
from matplotlib.pyplot import cla
from tensorflow import keras

class ImagePredictor:
    def __init__(self, model):
        pass
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, help="Path to input image")
    parser.add_argument("--mask_path", type=str, help="Path to output mask")
    parser.add_argument("--model", type=str, default = "unet", help="Name of the prediction model")
    parser.parse_args()

