# High Accuracy Leaf Segmentation

**Authors:** Rotem Green & Omer Ben-Ishay  
**Supervisors:** Alon Zvirin, Yaron Honen  
**Course:** 234329 Spring 2022/23

This repository contains the code for the project "High Accuracy Leaf Segmentation" which is part of the course 234329 - Image Processing and Analysis at the Technion.

The repository is based on: [Model-Leaf Repo](https://github.com/simonlsk/model-leaf)

## Project Focus

This project delves into the data augmentation techniques for a pre-existing system dedicated to instance segmentation of leaves within plant imagery. Instance segmentation entails the precise delineation and labeling of individual leaves, assigning a unique "instance" identifier to each. This capability underpins various applications within agriculture, such as automated leaf enumeration and disease identification.

## Projected Goals

This project aims to refine the data augmentation process within a plant image segmentation model. Our efforts will focus on three key areas:

1. **Improved Small Leaf Segmentation:** We will develop strategies to specifically address the challenges associated with segmenting smaller leaves.

2. **Precise Leaf Positional Augmentation:** We will focus on improving how leaves are positioned within the augmented images. Specifically we will use the original paper as a starting point for the “Triads Algorithm” resulting in a more realistic spatial distributions and mimicking natural growth patterns.

3. **Enhanced Realism with Alpha Blending:** To create more realistic synthetic images during data augmentation, we will incorporate alpha blending techniques. This will add a subtle blurring effect to the edges of the leaves, replicating the way leaves naturally overlap and obscure one another in real-world plant images.

By achieving these goals, we expect to improve the realism of the data augmentation process and therefore improve the accuracy of the leaf segmentation model. This will contribute to a more robust and reliable system for applications in agriculture and beyond.

## Code Overview

As our work is heavily related to the data augmentation process, most of our work can be seen in `LeafDataset.py`.

The algorithm which adds the alpha blending to the leaves can be found in `alpha_blend.py`.

Parameters that we added to the code in the configuration JSON files are:

- `small_leaf_percentile` - defines the percentile threshold for leaf’s height to be considered a small leaf. 
- `small_leaf_images_ratio` - defines the ratio of images that consists of small leaves only.
- `small_leaf_position_offset` - defines the offset of the small leaves from the center of the plant. 
- `small_leaves_in_image_ratio` - defines the ratio of small leaves in respect to total number of leaves per image.
- `use_triads` - defines whether to use the triads algorithm or not.

Functions for creating h5 files for the competition website can be found in `hdf5_utils.py`.

Scripts that enabled us to run train and inference on all folder types - `run_leaf_seg_train_all.bs`, `run_leaf_seg_infer_all.bs`.

Script for running infernce on A5 test set - `Infer_A5.py`.