from multiprocessing import Pool, cpu_count
from skimage.transform import rotate
from PIL import Image, ImageDraw
from pydoc import locate
import multiprocessing
from math import ceil
import numpy as np
import time
import tqdm
import os
multiprocessing.set_start_method('spawn', True)

COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)


def cut(args):
    # Reconstruct arguments
    annotation_path = args.path
    limit = args.limit
    output_dir = args.output
    width = args.normalize
    background = args.background
    class_name = args.adapter
    adapter_class = locate(class_name + '.' + class_name)
    should_rotate = args.rotate
    task_id = args.task

    # Track performance
    start = time.time()

    # Create cut sequence
    if task_id is not None:
        jobs = adapter_class(annotation_path, task_id, limit)
    else:
        jobs = adapter_class(annotation_path, limit)

    jobs_for_pool = []
    for leaf_annotation, image_path, i in jobs:
        job_pipe = [(image_from_annotation, (leaf_annotation, image_path))]
        if width is not None:
            job_pipe.append((resize_image, (width,)))
        if background != "transparent":
            job_pipe.append((apply_background, (background,)))
        if should_rotate:
            job_pipe.append((rotate_image, (jobs.get_point(i),)))

        job_pipe.append((save_image, (i, output_dir)))
        jobs_for_pool.append(job_pipe)

    total_jobs = len(jobs_for_pool) if limit is None else min(limit, len(jobs_for_pool))

    # Run cutting tasks asynchronously to exploit multiple CPU cores
    with Pool(cpu_count()) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(handle_job_pipe, jobs_for_pool), total=total_jobs):
            pass

    # Print performance
    print("Took {0:.2f} seconds".format(time.time() - start))


def image_from_tuple(tuple):
    return image_from_annotation(*tuple)


def handle_job_pipe(function_sequence):
    head_function, args = function_sequence[0]
    image = head_function(*args)
    for function, args in function_sequence[1:]:
        image = function(image, *args)


def apply_background(image, background):
    if image is None:
        return image
    if background == "original":
        return image.convert("RGB")

    color = COLOR_WHITE if background == "white" else COLOR_BLACK
    background = Image.new("RGB", image.size, color)
    background.paste(image, mask=image.split()[3])

    return background


def resize_image(image, width):
    if image is None:
        return image

    rgb_image = Image.fromarray(np.asarray(image)[:, :, :3], "RGB")
    mask_image = Image.fromarray(np.asarray(image)[:, :, 3], "L")
    ratio = rgb_image.size[0] / rgb_image.size[1]
    height = int(rgb_image.size[0] / ratio)
    rgb_image.thumbnail((width, height), Image.ANTIALIAS)
    mask_image.thumbnail((width, height), Image.NEAREST)
    rgb_image.putalpha(mask_image)

    return rgb_image


def save_image(image, suffix, dir_path):
    if image is None:
        return image

    path = "image_{}.png".format(suffix)
    os.makedirs(dir_path, exist_ok=True)
    if dir_path is not None:
        path = os.path.join(dir_path, path)
    try:
        image.save(path)
    except Exception as e:
        print("error occured while processing image {}".format(path))
        print(e)
        return None


def image_from_annotation(leaf_annotation, image_path):
    image = None
    try:
        image = Image.open(image_path).convert("RGBA")
    except Exception as e:
        print("error occured while opening file {}".format(image_path))
        print(e)

    # Recreate bbox from polygon
    vertices = np.array(leaf_annotation).reshape((-1, 2)).transpose()
    x, y = tuple(
        map(int,
            (min(vertices[0]), min(vertices[1]))
            )
    )
    x_right, y_bottom = tuple(
        map(lambda x: int(ceil(x)),
            (max(vertices[0]), max(vertices[1]))
            )
    )

    # Keep right and bottom values inside picture borders
    x = max(0, x)
    y = max(0, y)
    x_right = min(x_right, image.size[0])
    y_bottom = min(y_bottom, image.size[1])

    # Create new cropped image
    new_image_array = np.asarray(image)[y:y_bottom, x:x_right, :3]

    # Create and apply mask from polygon
    mask_image = Image.new('L', image.size)
    ImageDraw.Draw(mask_image).polygon(leaf_annotation, fill=255, outline=0)
    mask_image_array = np.expand_dims(np.asarray(mask_image)[y:y_bottom, x:x_right], axis=2)
    new_image_array = np.concatenate((new_image_array, mask_image_array), axis=2)

    try:
        new_image = Image.fromarray(new_image_array, "RGBA")
    except Exception as e:
        print("error occured while creating image from {}".format(image_path))
        return None

    return new_image


def rotate_image(image, points):
    """
    :param image:   Image object to rotate
    :param points:  Points of leaf edges. List of lists
    :return: rotated image
    """
    if points is None or len(points) < 2:
        return image

    top_x = points[0][0]
    top_y = points[0][1]
    bottom_x = points[1][0]
    bottom_y = points[1][1]

    delta_x = bottom_x - top_x
    delta_y = bottom_y - top_y

    angle_to_rotate_clockwise = np.degrees(np.arctan2([delta_x], [-delta_y]))[0]
    rotated_image = image.rotate(angle_to_rotate_clockwise, expand=True)

    return rotated_image
