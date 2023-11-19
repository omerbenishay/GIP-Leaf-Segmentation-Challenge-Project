from mrcnn.config import Config
import numpy as np

class LeafSegmentorConfig(Config):
    # Give the configuration a recognizable name
    NAME = "leaves"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 4  # AZ 1  2   8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 shape (leaves)

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    # IMAGE_MIN_DIM = 1024
    # IMAGE_MAX_DIM = 1024

    # Use smaller anchors because our image and objects are small
    # RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)    # AZ corn 2020_03_10
    # RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)  # AZ banana, avocado
    # RPN_ANCH#OR_SCALES = (32, 64, 128, 256, 512)


    #RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)  # AZ corn 2020_03_22
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)    # AZ avocado 2020_09_04
    #RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)  # AZ banana 601 plant masks


    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    #TRAIN_ROIS_PER_IMAGE = 62   #   32  16
    #TRAIN_ROIS_PER_IMAGE = 30   # AZ train 2019_11_09
    TRAIN_ROIS_PER_IMAGE = 62  # AZ train 2019_11_10

    # ROI_POSITIVE_RATIO = 66

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    # use small validation steps IF the epoch is small
    VALIDATION_STEPS = 500

    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])  # ImageNet
    #MEAN_PIXEL = np.array([0.0, 0.0, 0.0])    # tst
    #MEAN_PIXEL = np.array([105.9, 103.1, 93.8]) # sample from training set - banana
