import argparse
import os
from PIL import Image
import numpy as np

from tensorflow.keras.models import load_model

model_name = "unet_v4.tf"
image_file = "test_dataset/test/images/i_14896.png"
model = load_model(os.path.join("trained_models", model_name))


with Image.open(image_file) as f:
    img = np.array(f, dtype="float32")


