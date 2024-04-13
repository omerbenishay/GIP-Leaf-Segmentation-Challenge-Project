from PIL import Image
from tqdm import tqdm
import numpy as np
from questionary import select
from mrcnn import model as mrcnn_lib
import os
from Config import LeafSegmentorConfig

def get_inference_config(config):
    """
    Alters a config class to make it suited for inference
    Alterations are not mandatory for the inference to work
    :param config: the config class
    :return: a modified config class
    """
    config.IMAGES_PER_GPU = 1
    config.GPU_COUNT = 1
    return config()

def prompt_model(path):
    """
    Generate a correct .h5 model path. If the path is a directory, prompts the user for
    a correct path.
    This function recursively calls itself until a .h5 file is returned
    :param path: The specified path
    :return: the resulting path from prompt
    """
    if path.split('.')[-1] == "h5":
        return os.path.normpath(path)  # get rid of '/../' in path
    if os.path.isfile(path):
        return '..'

    choices = os.listdir(path) + ['..']
    my_question = 'Select the model you want to use for inference'
    next_dir = select(my_question, choices).ask()
    response = prompt_model(os.path.join(path, next_dir))
    return response

def generate_images(infer_path):
    """
    :param infer_path:
    :return: generator of image paths
    """
    if os.path.isdir(infer_path):
        for dir_path, _, files in os.walk(infer_path):
            for file in files:
                if file.split(".")[-1].lower() in ['jpg', 'jpeg', 'png', 'bmp']:
                    yield os.path.join(dir_path, file)
    else:
        if infer_path.split(".")[-1].lower() in ['jpg', 'jpeg', 'png', 'bmp']:
            return [os.path.join(infer_path)]

def save_masks(masks, output_dir, image_name):
    masks_shape = masks.shape
    image = np.zeros(masks_shape[:2], dtype='uint8')
    for i in range(masks_shape[-1]):
        mask = masks[..., i]
        image += mask.astype('uint8') * (i + 1)

    # my_cm = cm.get_cmap(COLOR_MAP, masks_shape[-1] + 1)
    # my_cm.set_bad(color='black')
    # image = np.ma.masked_where(image == 0, image)
    # image = np.uint8(my_cm(image) * 255)
    if image.ndim == 3:
        image = image[:, :, 0]
    mask_image_name = ".".join(image_name.split('.')[:-1]) + "_mask.png"
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    Image.fromarray(image).save(os.path.join(output_dir, mask_image_name))

a1_model_path = "/home/rotem.green/GIP-Leaf-Segmentation-Challenge-Project/models/A1/leaves20240406T1633/mask_rcnn_leaves.h5"
a2_model_path = "/home/rotem.green/GIP-Leaf-Segmentation-Challenge-Project/models/A2/leaves20240407T1031/mask_rcnn_leaves.h5"
a3_model_path = "/home/rotem.green/GIP-Leaf-Segmentation-Challenge-Project/models/A3/leaves20240408T0103/mask_rcnn_leaves.h5"
a4_model_path = "/home/rotem.green/GIP-Leaf-Segmentation-Challenge-Project/models/A4/leaves20240408T0823/mask_rcnn_leaves.h5"
inference_config = get_inference_config(LeafSegmentorConfig)

a1_model = a2_model = a3_model = a4_model = mrcnn_lib.MaskRCNN(mode="inference", config=inference_config, model_dir="outputs")

#A1
a1_model.load_weights(a1_model_path, by_name=True)
a1_model.set_log_dir()

a1_output_dir = a1_model.log_dir
os.makedirs(a1_output_dir, exist_ok=True)

#A2
a2_model.load_weights(a2_model_path, by_name=True)
a2_model.set_log_dir()

a2_output_dir = a2_model.log_dir
os.makedirs(a2_output_dir, exist_ok=True)

#A3
a3_model.load_weights(a3_model_path, by_name=True)
a3_model.set_log_dir()

a3_output_dir = a3_model.log_dir
os.makedirs(a3_output_dir, exist_ok=True)

#A4
a4_model.load_weights(a4_model_path, by_name=True)
a4_model.set_log_dir()

a4_output_dir = a4_model.log_dir
os.makedirs(a4_output_dir, exist_ok=True)

# Retrieve images
images = list(generate_images("test_set/A5"))

# Infer
inference_dict = {}
IoU_dict = {}
output_dir = a4_output_dir
for image_path in tqdm(images):
    model = None
    inference_dict[image_path] = []
    dir_name = image_path.split(os.path.sep)[-2]
    image_name = os.path.basename(image_path)
    image = Image.open(image_path)
    if image.size == (500, 530):
        model = a1_model
    if image.size == (530, 565):
        model = a2_model
    elif image.size == (2448, 2048):
        model = a3_model
    elif image.size == (441, 441):
        model = a4_model
    image = np.array(image)
    if image.shape[2] > 3:
        image = image[:,:,:3]
    r = model.detect([image])[0]
    save_masks(r['masks'], os.path.join(output_dir, dir_name) , image_name)

