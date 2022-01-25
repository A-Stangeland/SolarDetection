import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import join as join_path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPool2D, UpSampling2D, Concatenate, Input
from tensorflow.keras.models import Model
import tensorflow.keras.metrics as km
import tensorflow.keras.backend as K

from data_generation import SegmentationDataGenerator

def down_block(filters, conv_args, conv_in):
    x = Conv2D(filters, **conv_args)(conv_in)
    skip = Conv2D(filters, **conv_args)(x)
    down = MaxPool2D()(skip)
    return down, skip
    
def up_block(filters, conv_args, conv_in, skip):
    x = UpSampling2D()(conv_in)
    x = Concatenate(axis=-1)([conv_in, skip])
    x = Conv2D(filters, **conv_args)(x)
    up = Conv2D(filters, **conv_args)(x)
    return up

def iou(y_true, y_pred):
        a = y_true == 1
        b = y_pred > 0.5
        inter = tf.cast(tf.math.logical_and(a, b), "float32")
        union = tf.cast(tf.math.logical_or(a, b), "float32")
        return tf.reduce_sum(inter, axis=[1,2,3]) / tf.reduce_sum(union, axis=[1,2,3])

def create_unet(image_size=(128,128)):
    input_shape = (*image_size, 3)

    conv_args = dict(kernel_size=(3,3), padding="same", activation="relu")

    in_unet = Input(input_shape)
    down, skip1 = down_block(32, conv_args, in_unet)
    down, skip2 = down_block(64, conv_args, down)
    down, skip3 = down_block(128, conv_args, down)
    down, skip4 = down_block(256, conv_args, down)

    x = Conv2D(512, **conv_args)(down)
    x = Conv2D(512, **conv_args)(x)

    up = up_block(256, conv_args, x, skip4)
    up = up_block(128, conv_args, up, skip3)
    up = up_block(64, conv_args, up, skip2)
    up = up_block(32, conv_args, up, skip1)

    out_unet = Conv2D(1, kernel_size=(3,3), padding="same", activation="sigmoid")(up)

    unet = Model(in_unet, out_unet)
    return unet

def train_unet(dataset_path="dataset", batch_size=32, epochs=10, unet_save_name="unet.tf"):
    train_gen = SegmentationDataGenerator(join_path(dataset_path, "train"), batch_size=batch_size)
    test_gen = SegmentationDataGenerator(join_path(dataset_path, "test"), batch_size=batch_size)
    image_size = train_gen.get_image_size()
    
    unet = create_unet(image_size)
    unet.compile(
        loss="binary_crossentropy", 
        optimizer="rmsprop",
        metrics=["accuracy","precision", "recall"])
    
    training_history = unet.fit(train_gen, validation_data=test_gen, epochs=epochs)
    return unet, training_history

def main():
    dataset_path = "dataset"
    model_name = "unet"
    model, history = train_unet(dataset_path, unet_save_name=f"{model_name}.tf")
    
    if not os.path.exists("trained_models"):
            os.makedirs("trained_models")
    model.save(os.path.join("trained_models", f"{model_name}.tf"))

if __name__ == "__main__":
    main()
