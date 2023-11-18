
import math
import cv2
import numpy as np
from PIL import Image


def rotate_bound(image, angle):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    return cv2.warpAffine(image, M, (nW, nH))


def add_image(img1, img2, x_center, y_center, x_scale, y_scale, angle):
    img2 = img2.resize((int(x_scale * img2.size[0]), int(y_scale * img2.size[1])), resample=Image.BICUBIC) # Image.ANTIALIAS
    img2 = img2.rotate(angle, resample=Image.BICUBIC, expand=True)

    rows, cols, channels = np.asarray(img2).shape
    x_from = x_center - math.floor(cols / 2.)
    y_from = y_center - math.floor(rows / 2.)

    img1.paste(img2, (x_from, y_from), img2)
    # tmp_mask = image_to_mask(img2)
    # tmp_mask = Image.fromarray(tmp_mask)
    # img1.paste(img2, (x_from, y_from), tmp_mask)

    return img1

def image_to_mask(image):
    # AZ TODO: check if OK when image is 2D (grayscale)
    img_sum = np.sum(image, axis=-1)
    mask = img_sum > 0
    #se = scipy.ndimage.generate_binary_structure(2, 1)
    #mask = scipy.ndimage.binary_erosion(mask, structure=se, iterations = 2)
    return mask

def mask_to_image(mask):
    x, y, z = mask.shape
    image = np.zeros((x, y), dtype=np.uint8)
    for i in range(0, z):
        mask_color = int(((i + 1) / z) * 255)
        image += mask[:, :, i] * np.cast[np.uint8](mask_color)
    return image

def add_image_without_transparency(img1, img2, x_center, y_center, x_scale, y_scale, angle):
    img2 = cv2.resize(img2, None, fx=x_scale, fy=y_scale, interpolation=cv2.INTER_CUBIC)

    img2 = rotate_bound(img2, 360 - angle)

    rows, cols, channels = img2.shape
    x_from = x_center - math.floor(cols / 2.)
    x_to = x_center + math.ceil(cols / 2.)
    y_from = y_center - math.floor(rows / 2.)
    y_to = y_center + math.ceil(rows / 2.)

    y_max, x_max, _ = img1.shape

    if x_from < 0:
        img2 = img2[:, -x_from:]
        x_from = 0
    if x_to >= x_max:
        img2 = img2[:, :-(x_to - x_max + 1)]
        x_to = x_max - 1
    if y_from < 0:
        img2 = img2[-y_from:, :]
        y_from = 0
    if y_to >= y_max:
        img2 = img2[:-(y_to - y_max + 1), :]
        y_to = y_max - 1

    roi = img1[y_from:y_to, x_from:x_to]

    img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    img2_fg = cv2.bitwise_and(img2, img2, mask=mask)
    #dst = cv2.add(img1_bg, img2_fg[:, :, :])  # AZ (remove alpha)
    dst = cv2.add(img1_bg, img2_fg[:, :, 0:3])
    img1[y_from:y_to, x_from:x_to] = dst
    return img1
