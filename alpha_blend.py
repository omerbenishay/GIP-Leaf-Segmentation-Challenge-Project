import numpy as np
import os
import argparse
from PIL import Image
from scipy.ndimage import distance_transform_edt

parser = argparse.ArgumentParser(description='Alpha blending')
parser.add_argument('-A', '--A_folder', choices=['all', '1', '2', '3', '4'],
                    help='Folder to apply alpha blending on', default='all')
parser.add_argument('-D', '--distance_threshold', type=float,
                    help='Distance for thresholding alpha values', default='0.1')

_CLEAN_LEAVES_PATH = '/home/rotem.green/GIP-Leaf-Segmentation-Challenge-Project/datasets/06 - Leaves/A{folder_num}_clean/'
_OUTPUT_PATH = '/home/rotem.green/GIP-Leaf-Segmentation-Challenge-Project/datasets/06 - Leaves/A{folder_num}_alpha_{distance_threshold}/'


def apply_alpha_blend(image_path: str, distance_threshold: float = 0.1) -> Image:
    img = Image.open(image_path)
    img_array = np.array(img)

    leaf_mask = img_array[:, :, 3] > 0
    img_array = img_array[:, :, :3]

    img_height, img_width = leaf_mask.shape
    # Each pixel value is the sum of the row (leaf's width) it belongs to
    width_arr = np.repeat(leaf_mask.sum(axis=1), img_width).reshape(img_height, img_width)

    # Apply distance transform divided by width
    alpha = distance_transform_edt(leaf_mask) / width_arr
    # Fill nans with 0
    alpha = np.nan_to_num(alpha)
    # Normalize alpha values to [0, 1]
    alpha = alpha / alpha.max()
    # Threshold alpha values
    alpha[alpha > distance_threshold] = 1
    # Convert alpha values to uint8
    alpha = (255 * alpha).astype(np.uint8)
    alpha = alpha[:, :, np.newaxis]

    # Create RGBA image with alpha blending
    rgba_img = Image.fromarray(np.concatenate([img_array, alpha], axis=2), mode='RGBA')

    return rgba_img


def alpha_blend(a_folder, distance_threshold):
    # Define input and output directories
    input_dir = _CLEAN_LEAVES_PATH.format(folder_num=a_folder)
    output_dir = _OUTPUT_PATH.format(folder_num=a_folder, distance_threshold=distance_threshold)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through images in the input directory
    for filename in os.listdir(input_dir):
        image_path = os.path.join(input_dir, filename)
        new_image = apply_alpha_blend(image_path, distance_threshold)
        new_image.save(os.path.join(output_dir, filename))

    print(f"Alpha blending applied for folder {a_folder} and D={distance_threshold}.")


def main():
    # read args
    args = parser.parse_args()
    a_folder = args.A_folder
    distance_threshold = args.distance_threshold

    if a_folder == 'all':
        for i in range(1, 5):
            alpha_blend(i, distance_threshold)
    else:
        alpha_blend(a_folder, distance_threshold)

if __name__ == '__main__':
    main()

