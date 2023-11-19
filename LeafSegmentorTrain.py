from pydoc import locate
import json
from Config import LeafSegmentorConfig
from mrcnn import utils
import numpy as np
from PIL import Image
import os
from DatasetUtils import mask_to_image
from tqdm import tqdm
from LeafSegmentorUtils import get_clean_dict_from_class, add_metadata_dict_to_h5

CONFIG_METADATA_NAME = 'Config'


def train(args):
    from mrcnn.model import MaskRCNN
    # Retrieve arguments
    output = args.output
    samples_number = args.dataset_keep
    preview_only = args.preview_only
    dataset_config_path = args.dataset_config_file
    epochs = args.epochs
    steps_per_epoch = args.steps_per_epoch
    layers = args.layers
    pretrain = args.pretrain
    Afolder = args.Afolder
    leaf_type = args.leaf_type
    
    output = os.path.join(output, "A" + Afolder)
    # Create config class
    train_config = LeafSegmentorConfig()

    # Create model
    if steps_per_epoch > 0:
        train_config.STEPS_PER_EPOCH = steps_per_epoch

    if not os.path.exists(output):
        os.makedirs(output, exist_ok=True)
    model = MaskRCNN(mode="training", config=train_config, model_dir=output)

    # Create dataset
    with open(dataset_config_path) as dataset_config_file:
        dataset_config = json.load(dataset_config_file)
    dataset_class = dataset_config["dataset_module"] + "." + dataset_config["dataset_class"]
    dataset_class = locate(dataset_class)

    dataset_config = dataset_config["config"]
    dataset_train = dataset_class.from_config(dataset_config["train"], train_config.IMAGE_SHAPE[0], train_config.IMAGE_SHAPE[1], Afolder, leaf_type)
    dataset_valid = dataset_class.from_config(dataset_config["valid"], train_config.IMAGE_SHAPE[0], train_config.IMAGE_SHAPE[1], Afolder, leaf_type)

    # Save train samples
    if samples_number > 0:
        samples_output_dir = os.path.join(model.log_dir, "samples")
        save_samples(dataset_valid, samples_number, path=samples_output_dir)

    if preview_only:
        return  # finish here

    # Start training from COCO or from previously trained model
    if pretrain == "COCO":
        load_coco_weights(model)
    else:
        model.load_weights(pretrain, by_name=True)

    # Start train
    model.train(dataset_train, dataset_valid, learning_rate=train_config.LEARNING_RATE, epochs=epochs, layers=layers)

    # Add metadata to model files (.h5)
    add_config_to_model(model.log_dir)


def apply_class_configuration(class_instance, config_dict):
    for key, value in config_dict.items():
        setattr(class_instance, key, value)


def save_samples(dataset, n=10, path="./"):
    """
    :params dataset: dataset instance
    :param n: number of samples to create
    :param path: absolute path to store the samples at
    """
    os.makedirs(path, exist_ok=True)
        
    image_ids = np.random.choice(dataset.image_ids, n)
    print("Saving {} samples to {}".format(n, path))
    for image_id in tqdm(image_ids):
        image = dataset.load_image(image_id)
        mask, _ = dataset.load_mask(image_id)
        image_path = os.path.join(path, "{}_image.png".format(image_id))
        Image.fromarray(image).save(image_path)
        mask_path = os.path.join(path, "{}_msk.png".format(image_id))
        Image.fromarray(mask_to_image(mask)).convert("RGB").save(mask_path)


def load_coco_weights(model, dir_path="models"):
    """
    :param model:       MaskRCNN model instance to load model to
    :param dir_path:    Directory to check for existing coco pretrain file
                        or to download it to
    """
    pretrain_file_path = os.path.join(dir_path, "mask_rcnn_coco.h5")
    if not os.path.exists(pretrain_file_path):
        os.makedirs(os.path.dirname(pretrain_file_path), exist_ok=True)
        utils.download_trained_weights(pretrain_file_path)

    model.load_weights(pretrain_file_path, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])


def add_config_to_model(model_path):
    config_dict = get_clean_dict_from_class(LeafSegmentorConfig)
    # iterate over .h5 files
    for _, _, files in os.walk(model_path):
        for file in files:
            if file.endswith('.h5'):
                add_metadata_dict_to_h5(os.path.join(model_path, file), CONFIG_METADATA_NAME, config_dict)
