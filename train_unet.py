import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import shutil
from argparse import ArgumentParser

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPool2D, UpSampling2D, Concatenate, Input
from tensorflow.keras.models import Model, load_model
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
    x = Concatenate(axis=-1)([x, skip])
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

def train_unet(
        dataset_path="dataset", 
        batch_size=32, 
        epochs=10, 
        loss="binary_crossentropy", 
        optimizer="adam"):
    """Train a unet model on the test part of the dataset and evaluate on the validation part."""
    train_gen = SegmentationDataGenerator(os.path.join(dataset_path, "train"), batch_size=batch_size)
    test_gen = SegmentationDataGenerator(os.path.join(dataset_path, "test"), batch_size=batch_size)
    image_size = train_gen.get_image_size()
    
    # Defining the evaluation metrics
    eval_metrics = [
        km.BinaryAccuracy(name="accuracy"),
        km.AUC(name="PR_AUC", curve="PR"),
        km.Precision(name="precision"),
        km.Recall(name="recall")
    ]

    unet = create_unet(image_size)
    unet.compile(
        loss = loss, 
        optimizer = optimizer,
        metrics=eval_metrics)
    
    training_history = unet.fit(train_gen, validation_data=test_gen, epochs=epochs)
    training_history_df = pd.DataFrame(**training_history.history)
    unet.evaluate(test_gen)
    return unet, training_history_df

def evaluate_model(model, dataset_path):
    """Evaluate the model on the given dataset."""
    eval_gen = SegmentationDataGenerator(dataset_path)
    model.evaluate(eval_gen)

def validate_model_path(model_name, model_save_path):
    """Checks if the model already exists and is so, asks the user if they want to overwrite."""
    name_validated = False
    while not name_validated:
        model_path = os.path.join(model_save_path, model_name)
        model_exists = os.path.exists(model_path)
        if model_exists:
            overwrite = input(f"There is already a model named {model_name}. Do you wish to overwrite?([y]/n)")
            if overwrite.lower() in ["yes", "y", ""]:
                shutil.rmtree(model_path)
                os.makedirs(model_path)
                name_validated = True
            elif overwrite.lower() in ["no", "n"]:
                model_name = input("Please provide a new model name:")
            else:
                raise ValueError("Invaid input. Aborting.")
        else:
            os.makedirs(model_path)
            name_validated = True
    return model_name, model_path


def main():
    """Either train a new model or evaluates an existing model depending on the --mode argument.

    If in train mode:
        First verifies that the provided model name han not already been used.
        If a model with the name exists, the user is asked if the want to overwrite or provide a new name.
        The model is then trained using the parameters provided in the train_config.json file.
        Finally, both the model and the training history is saved in a directory with the model name.
    If in eval mode:
        The model is loaded and evaluated on the dataset at eval_dataset_path provided in the train_config.json file.
    """
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default = "unet", help="Name of the model")
    parser.add_argument("--mode", type=str, default = "train", help="Train/eval mode")
    args = parser.parse_args()
    model_name = args.model_name
    with open("train_config.json", mode="r") as f:
        config = json.load(f)
    model_save_path = config["model_save_path"]
    if args.mode.lower() in ["t", "train"]:
        model_name, model_path = validate_model_path(model_name, model_save_path)
        training_args = ["dataset_path", "batch_size", "epochs", "loss", "optimizer"]
        training_parameters = {key: config[key] for key in training_args}
        model, training_history = train_unet(**training_parameters)
        model.save(os.path.join(model_path, f"{model_name}.tf"))
        training_history.to_csv(os.path.join(model_path, f"{model_name}_history.csv"), index=False)
    elif args.mode.lower() in ["e", "eval", "evaluate"]:
        model_path = os.path.join(model_save_path, model_name)
        if not os.path.exists(model_path):
            raise ValueError(f"The model {model_name} does not exist at the location: {model_save_path}")
        
        model = load_model(os.path.join(model_path, f"{model_name}.tf"))
        evaluate_model(model, args["eval_dataset_path"])
        

if __name__ == "__main__":
    main()
