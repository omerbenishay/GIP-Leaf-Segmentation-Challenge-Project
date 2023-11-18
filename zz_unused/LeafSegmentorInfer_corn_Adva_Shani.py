from LeafSegmentorUtils import prompt_model
from pycocotools import mask as mask_tools
from Config import LeafSegmentorConfig
import matplotlib.pyplot as plt
from mrcnn import visualize
from skimage import measure
from matplotlib import cm
from pydoc import locate
from tqdm import tqdm
from PIL import Image
import numpy as np
import json
import os
import math
from mrcnn import utils

COLOR_MAP = "Blues"
CONTOUR_FILE_NAME = "contours.json"


def appending(a, b, c):
    res = []
    N = a.shape[0]
    for i in range(N):
        res.append(a[i])
    for i in range(b.shape[0]):
        res.append(b[i])
    for i in range(c.shape[0]):
        res.append(c[i])
    return res


def convert_cropped_mask(cropped_mask, bbox, image_shape, y1_new, x1_new, y2_new, x2_new):
    # Convert cropped mask to full size mask
    y1, x1, y2, x2 = bbox
    full_mask = np.zeros(image_shape[:2], dtype=np.bool)
    full_mask[y1_new:y2_new, x1_new:x2_new] = cropped_mask[y1:y2, x1:x2]
    return full_mask


def adjust_boxes(boxes, y_offset, x_offset):
    res = boxes
    for i in range(boxes.shape[0]):
        res[i][0] = boxes[i][0] + y_offset
        res[i][1] = boxes[i][1] + x_offset
        res[i][2] = boxes[i][2] + y_offset
        res[i][3] = boxes[i][3] + x_offset
    return res


def infer(args):
    from mrcnn import model as mrcnn_lib
    infer_path = args.path
    output = args.output
    do_pictures = not args.no_pictures
    do_contours = not args.no_contours
    model_path = args.model
    should_save_masks = not args.no_masks
    gt_adapter = args.gt
    task_id = args.task
    compare_to_gt = gt_adapter is not None

    # Retrieve images
    images = list(generate_images(infer_path))

    # Retrieve gt masks
    if compare_to_gt:
        adapter_class = locate(gt_adapter + '.' + gt_adapter)
        gt_path = infer_path
        gt_annotation_map = generate_annotation_map(adapter_class, gt_path, images, task_id)

    # Retrieve model path
    model_path = prompt_model(model_path)

    # Load model
    inference_config = get_inference_config(LeafSegmentorConfig)
    if not os.path.exists(output):
        os.makedirs(output, exist_ok=True)
    model = mrcnn_lib.MaskRCNN(mode="inference", config=inference_config, model_dir=output)
    model.load_weights(model_path, by_name=True)
    model.set_log_dir()

    output_dir = model.log_dir
    os.makedirs(output_dir, exist_ok=True)

    # Infer
    inference_dict = {}
    IoU_dict = {}
    crop_min_height = 400
    crop_min_width = 1550
    for image_path in tqdm(images):
        inference_dict[image_path] = []
        image_name = os.path.basename(image_path)
        image = np.array(Image.open(image_path))
        r = model.detect([image])[0]

        # A&S #
        boxes = r['rois']
        N = boxes.shape[0]
        xs, ys = [], []
        for i in range(N):
            y1, x1, y2, x2 = boxes[i]
            xs.extend([x1, x2])
            ys.extend([y1, y2])

        x_min = min(xs)
        x_max = max(xs)
        y_min = min(ys)
        y_max = max(ys)

        if x_max - x_min < crop_min_width:
            x_max = x_min + crop_min_width

        if y_max - y_min < crop_min_height:
            y_max = y_min + crop_min_height

        crop_len = math.floor((y_max - y_min)/3)
        adding = math.floor((x_max - x_min)/3)

        tmp_image = Image.open(image_path)
        width, height = tmp_image.size

        x1 = max(0, x_min - adding, math.floor(0.1*width))
        y1 = max(0, y_min - int(adding/2))
        x2 = min(math.floor(0.7*width), x_max + adding)
        y2 = min(math.floor(0.85*height), y_min + crop_len)

        part1 = tmp_image.crop((x1, y1, x2, y2))
        part2 = tmp_image.crop((x1, y2, x2, min(image.shape[1], y_min + 2*crop_len)))
        part3 = tmp_image.crop((x1, min(image.shape[1], y_min + 2*crop_len),
                                x2, min(image.shape[1], y_min + 3*crop_len + int(adding/2))))

        part1_np = np.array(part1)
        part2_np = np.array(part2)
        part3_np = np.array(part3)

        r1 = model.detect([part1_np])[0]
        r2 = model.detect([part2_np])[0]
        r3 = model.detect([part3_np])[0]

        r['class_ids'] = np.array(appending(r1['class_ids'], r2['class_ids'], r3['class_ids']))

        r['scores'] = np.array(appending(r1['scores'], r2['scores'], r3['scores']))

        masks1 = r1['masks']
        masks2 = r2['masks']
        masks3 = r3['masks']

        boxes1 = r1['rois']
        boxes2 = r2['rois']
        boxes3 = r3['rois']

        full_masks = []
        for i in range(boxes1.shape[0]):
            full_mask = convert_cropped_mask(masks1[:, :, i], boxes1[i], image.shape,
                                             boxes1[i][0] + y1, boxes1[i][1] + x1,
                                             boxes1[i][2] + y1, boxes1[i][3] + x1)
            full_masks.append(full_mask)
        for i in range(boxes2.shape[0]):
            full_mask = convert_cropped_mask(masks2[:, :, i], boxes2[i], image.shape,
                                             boxes2[i][0] + y2,  boxes2[i][1] + x1,
                                             boxes2[i][2] + y2, boxes2[i][3] + x1)
            full_masks.append(full_mask)
        for i in range(boxes3.shape[0]):
            full_mask = convert_cropped_mask(masks3[:, :, i], boxes3[i], image.shape,
                                             boxes3[i][0] + min(image.shape[1], y_min + 2*crop_len), boxes3[i][1] + x1,
                                             boxes3[i][2] + min(image.shape[1], y_min + 2*crop_len), boxes3[i][3] + x1)
            full_masks.append(full_mask)

        full_masks = np.stack(full_masks, axis=-1) \
            if full_masks else np.empty(image.shape[:2] + (0,))

        r['masks'] = full_masks

        boxes1 = adjust_boxes(boxes1, y1, x1)
        boxes2 = adjust_boxes(boxes2, y2, x1)
        boxes3 = adjust_boxes(boxes3, min(image.shape[1], y_min + 2*crop_len), x1)

        r['rois'] = np.array(appending(boxes1, boxes2, boxes3))
        # A&S #

        if should_save_masks:
            save_masks(r['masks'], output_dir, image_name)

        if do_pictures:
            output_file_path = os.path.join(output_dir, image_name)
            _, ax = plt.subplots(1, figsize=(16,16))
            visualize.save_instances(image, r['rois'], r['masks'], r['class_ids'],
                                     ['BG', 'leave'], r['scores'], save_to=output_file_path, ax=ax)

        if do_contours:
            inference_dict[image_path], txt_contours = get_contours(r)

            for i, leaf_contour in enumerate(txt_contours):
                for j, polygon_contour in enumerate(leaf_contour):
                    contour_file_name = os.path.join(output_dir, os.path.splitext(image_name)[0]) + \
                                        "_" + str(i).zfill(3) + "_" + str(j) + ".txt"
                    np.savetxt(contour_file_name, polygon_contour, fmt='%.1f', delimiter=' , ')

        if compare_to_gt:
            gt_masks = get_all_masks(gt_annotation_map[image_path], image_path)
            gt_image_name = ".".join(image_name.split(".")[:-1]) + "_GT.png"
            save_masks(gt_masks, output_dir, gt_image_name)
            IoU_dict[image_path] = calculate_iou(image_name, r['masks'], gt_masks)

            # A&S #
            # showing diff between gt and the masks we got on the same image
            """gt_boxes = []
            gt_class_ids = []
            for annotation in gt_annotation_map[image_path]:
                # Recreate bbox from polygon
                vertices = np.array(annotation).reshape((-1, 2)).transpose()
                x, y = tuple(
                    map(int,
                        (min(vertices[0]), min(vertices[1]))
                        )
                )
                x_right, y_bottom = tuple(
                    map(lambda x: int(math.ceil(x)),
                        (max(vertices[0]), max(vertices[1]))
                        )
                )

                # Keep right and bottom values inside picture borders
                x = max(0, x)
                y = max(0, y)
                x_right = min(x_right, image.shape[0])
                y_bottom = min(y_bottom, image.shape[1])

                gt_boxes.append((y, x, y_bottom, x_right))

                gt_class_ids = np.ones(gt_masks.shape[2], dtype=np.int32)

            output_file_path = os.path.join(output_dir, image_name.replace(".bmp", "_gt.bmp"))
            _, ax = plt.subplots(1, figsize=(16, 16))
            visualize.save_differences(image, np.array(gt_boxes), gt_class_ids, gt_masks,
                                       r['rois'], r['class_ids'], r['scores'], r['masks'],
                                       ['BG', 'leave'], title="", ax=ax,
                                       iou_threshold=0, score_threshold=0, save_to=output_file_path)"""

            # A&S #

    if do_contours:
        with open(os.path.join(output_dir, CONTOUR_FILE_NAME), 'w') as f:
            f.write(json.dumps(inference_dict, indent=2))

    if compare_to_gt:
        total_score = sum(IoU_dict.values()) / len(IoU_dict)
        print("average IoU scores: " + str(total_score))


def save_masks(masks, output_dir, image_name):
    masks_shape = masks.shape
    image = np.zeros(masks_shape[:2], dtype='uint8')
    for i in range(masks_shape[-1]):
        mask = masks[..., i]
        image += mask.astype('uint8') * (i + 1)

    my_cm = cm.get_cmap(COLOR_MAP, masks_shape[-1] + 1)
    my_cm.set_bad(color='black')
    image = np.ma.masked_where(image == 0, image)
    image = np.uint8(my_cm(image) * 255)
    mask_image_name = ".".join(image_name.split('.')[:-1]) + "_mask.png"
    Image.fromarray(image).save(os.path.join(output_dir, mask_image_name))


def get_contours(r):
    contours = {}
    orig_contours = []
    for i in range(r['masks'].shape[-1]):
        mask = r['masks'][..., i]
        # A mask might have multiple polygons
        mask_contours = measure.find_contours(mask, 0.5)
        # reshape in numpy then convert to list
        contours["leaf_{}".format(i)] = [np.reshape(c, (-1,)).tolist() for c in mask_contours]
        orig_contours.append(mask_contours)

    return contours, orig_contours


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


def generate_annotation_map(adapter_class, gt_dir, images, task_id=None):
    """
    :param adapter_class:
    :param images: list or generator of images paths
    :param gt_dir: directory that contains the annotation files
    :param task_id: argument to adapter
    :return: dictionary {"image_path": list(leave_polygons)}
    """
    images = list(images)
    if task_id is not None:
        annotations = adapter_class(gt_dir, task_id)
    else:
        annotations = adapter_class(gt_dir)

    leaves_map = {}
    for annotation, image_path, i in annotations:
        if image_path not in images:
            continue

        if leaves_map.get(image_path, None) is None:
            leaves_map[image_path] = []

        leaves_map[image_path].append(annotation)

    return leaves_map


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


GROUND_TRUTH_MIN_SIZE_COEFF = 0.05  # 0.03    0.05


def calculate_iou(image_name, detected_masks, gt_masks, single_mask=True):
    # AZ start validation of single image
    # TODO - log/results file

    # get ground truth masks for this image
    # note: this should be done only once for each validation image (if train, do it once at the beginning, not after each epoch).
    # image_name_prefix = image_name.split(".")[0] + "_GT_"
    # num_gt_masks = 0
    # h = detected_masks.shape[0]
    # w = detected_masks.shape[1]
    # gt_min_size = GROUND_TRUTH_MIN_SIZE_COEFF * GROUND_TRUTH_MIN_SIZE_COEFF * h * w
    #
    # gt_file_names = []
    # for root, dirs, files in os.walk(ground_truth_dir):
    #     for file in files:
    #         if file.startswith(image_name_prefix):
    #             # read GT file, and use the GT only if num_pixels in mask > Threshold
    #             tmp = np.array(Image.open(ground_truth_dir + file))
    #             tmp_size = np.count_nonzero(tmp)
    #             if tmp_size > gt_min_size:
    #                 gt_file_names.append(file)
    #                 num_gt_masks = num_gt_masks + 1
    #                 print(file)

    # gt_masks = np.zeros([h,w,num_gt_masks])
    num_gt_masks = gt_masks.shape[-1]

    # for i in range(num_gt_masks):
    #     curr_gt_file = ground_truth_dir + gt_file_names[i]
    #     curr_mask = np.array(Image.open(curr_gt_file))
    #     gt_masks[:,:,i] = curr_mask
    # create empty IoU matrix M (num_ground_truth_masks x num detected_masks)
    # note: if validation during training - this should be done after each epoch.
    num_of_detected_masks = detected_masks.shape[2]
    #print("num_of_detected_masks = ", detected_masks.shape[2])
    all_iou = np.zeros(shape=[num_gt_masks, num_of_detected_masks])

    # fill IoU matrix
    # for each mask m1 in ground truth
    #   for each mask m2 in detected
    #       M(m1,m2) = IoU(m1,m2)
    for i in range(num_gt_masks):
        mask_i = gt_masks[:,:,i]
        for j in range(num_of_detected_masks):
            mask_j = detected_masks[:,:,j]
            intersection = np.logical_and(mask_i,mask_j)
            union = np.logical_or(mask_i,mask_j)
            numI = np.count_nonzero(intersection)
            numU = np.count_nonzero(union)
            all_iou[i,j] = numI/numU

    # calculate total (or average) IoU
    curr_score = 0
    for i in range(num_gt_masks):
        # find max value and indices of max value
        max_iou = np.amax(all_iou)
        curr_score = curr_score + max_iou
        max_idx = np.argmax(all_iou)
        max_idx_row, max_idx_col = divmod(max_idx, all_iou.shape[1])

        # remove row/col of max value (set zeros)
        for j in range(all_iou.shape[1]):
            all_iou[max_idx_row,j] = 0
        for j in range(all_iou.shape[0]):
            all_iou[j,max_idx_col] = 0

    if num_gt_masks > 0:
        curr_score = curr_score / num_gt_masks
    else:
        curr_score = 1

    print("IoU score: " + str(curr_score))
    return curr_score
    # AZ end validation of single image


def get_all_masks(leaves_list, image_path):
    """Each leaf is a tuple (leaf_annotation, image_path, i)"""
    with Image.open(image_path) as im:
        image_width, image_height = im.size
    masks = np.empty(shape=(image_height, image_width, len(leaves_list)))

    for i, polygon_list in enumerate(leaves_list):
        # A single leaf might have multiple polygons but in can the adapter doesn't support it
        if not isinstance(polygon_list[0], list):
            polygon_list = [polygon_list]
        masks[..., i] = get_mask_from_list(polygon_list, image_width, image_height)

    return masks


def get_mask_from_list(xy_polygons, width, height):
    """
    :param xy_polygons: list of polygons
    :type xy_polygons: list(list(float))
    :param width: image width
    :param height: image height
    :return: a mask of shape (width, height)
    :rtype: numpy array of float, or None if no mask is exists
    """
    rle = ann_to_rle(xy_polygons, height, width)
    m = mask_tools.decode(rle)
    # Some objects are so small that they're less than 1 pixel area
    # and end up rounded out. Skip those objects.
    if m.max() < 1:
        return None

    return m


def ann_to_rle(segm, height, width):
    """
    Convert annotation which can be polygons, uncompressed RLE to RLE.
    :return: binary mask (numpy 2D array)
    """
    if isinstance(segm, list):
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = mask_tools.frPyObjects(segm, height, width)
        rle = mask_tools.merge(rles)

    return rle
