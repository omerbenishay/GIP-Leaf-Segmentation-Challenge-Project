
import os
import json
import numpy as np
import skimage
#from skimage import draw, img_as_ubyte
#from skimage.io import imread
import skimage.morphology
#from skimage.morphology import disk
#from skimage.transform import rotate
import math

# notes:
# for annotation task_ID_XXX, use appropriate values of TASK_ID_XXX , IMG_DIR_XXX , JSON_DIR_XXX , OUT_DIR_XXX


# tasks IDs, directories of image and json files, directory of output files
# avocado plantlets

TASK_ID_170 = 170
IMG_DIR_170 = 'C:/alon/Phenomics_tst_images_jsons/task_170/'
JSON_DIR_170 = 'C:/alon/Phenomics_tst_images_jsons/task_170/'
OUT_DIR_170 = 'C:/alon/agriculture/avocado/ann_170_leaves_A/'

TASK_ID_171 = 171
IMG_DIR_171 = 'C:/alon/Phenomics_tst_images_jsons/task_171/'
JSON_DIR_171 = 'C:/alon/Phenomics_tst_images_jsons/task_171/'
OUT_DIR_171 = 'C:/alon/agriculture/avocado/ann_171_leaves_A/'

IMG_DIR_AVOCADO_ABCD = 'C:/alon/Phenomics_tst_images_jsons/task_avocado_all/'
JSON_DIR_AVOCADO_ABCD = 'C:/alon/Phenomics_tst_images_jsons/task_avocado_all/'
OUT_DIR_AVOCADO_ABCD = 'C:/alon/agriculture/avocado/avocado_all_A/'


# maize plantlets
TASK_ID_172 = 172
IMG_DIR_172 = 'C:/alon/Phenomics_tst_images_jsons/task_172/'
JSON_DIR_172 = 'C:/alon/Phenomics_tst_images_jsons/task_172/'
OUT_DIR_172 = 'C:/alon/agriculture/corn_ann_172/tst_leaves_aligned_S2/'


# banana plantlets
TASK_ID_103 = 103
TASK_ID_105 = 105
TASK_ID_107 = 107
IMG_DIR_103 = 'C:/alon/Phenomics_tst_images_jsons/tasks_103_105_107_as_png/'
JSON_DIR_103 = 'C:/alon/Phenomics_tst_images_jsons/tasks_103_105_107_as_png/'
OUT_DIR_103 = 'C:/alon/agriculture/banana_intel_ABCD/tst_leaves_aligned/'


# valid image file extensions
IMG_FILE_EXTENSIONS = ['.bmp', '.jpg', '.jpeg', '.png', '.tiff']

# ID's of relevant dictionary records from Phenomics dictionary
# dictionary records [1967 .. 1976] correspond to group [1st .. 10th]
# dictionary records [2044 .. 2063] correspond to group [11th .. 30th]
DIC_GROUP_IDS = [i for i in range(1967,1977)] + [i for i in range(2044,2064)]
# start/end points of a leaf
DIC_POSITION_START = 1962
DIC_POSITION_END = 1963
DIC_POSITION_UPPER = 1964   # same as DIC_POSITION_END
DIC_POSITION_LOWER = 1966   # same as DIC_POSITION_START
DIC_POSITION_BASAL = 1993   # same as DIC_POSITION_START

# minimum number of points per polygon
MIN_POLYGON_POINTS = 3
# maximum number of leaves per plant
MAX_LEAVES_PER_PLANT = len(DIC_GROUP_IDS)

# parameters for output leaf images
DO_ERODE = 0            # specify for small erosion of leaf + mask
se = skimage.morphology.disk(radius=5)     # erosion kernel
DO_RESIZE = 0           # specify for leaf resize
RESIZE_SCALE = 0.15     # resize scale
DO_FIXED_OUTSIZE = 0    # specify for fixing max(out_image_height, out_image_width)
FIXED_OUTSIZE = 128     # 128  256  512
DO_ALPHA_LEAVES = 1     # write alpha channel in output file


def parse_json_task_171(json_data, task_ids):
    # read relevant info from json data
    #   type: polygon/point
    #   dictionary records: leaf ordinality (1st, 2nd, 3rd, ...) and start/end point of leaf
    #   coordinates : x,y coords of leaf contours or leaf extreme points

    leaf_data = []
    for key in json_data:
        # note: these jsons have a single key, all data contained in the value of a single key.
        json_data = json_data[key]

    # extract data type, dictionary records, and coordinate values
    num_data = len(json_data)
    for i in range(num_data):
        curr = json_data[i]
        is_deleted = curr.get('deleted')
        ann_task_id = curr.get('annotation_task_id')
        if is_deleted != 0:
            # discard this annotation if it was deleted
            continue
        if not (ann_task_id in task_ids):
            # discard this annotation if it doesn't belong to the scecified task Id's
            continue
        ann_type = curr.get('annotation_type')
        if not(ann_type == 'polygon' or ann_type == 'point'):
            # discard this annotation if it is not listed as a point or poygon
            continue
        dic_records = curr.get('annotation_dictionary_records')
        ann_records = []
        num_dic_records = len(dic_records)
        for j in range(num_dic_records):
            # get all dictionary records for this annotation
            ann_records.append(dic_records[j].get('record_id'))
        if len(ann_records) == 0:
            # discard this annotation if it has no dictionary records
            continue
        ann_vals = curr.get('annotations')
        if ann_type == 'polygon':
            # get list of points  (x,y pairs) defining polygon-contour of an object
            coords = [[ann['x'], ann['y']] for ann in ann_vals]
        elif ann_type == 'point':
            # get single point (single x,y pair)
            coords = [ann_vals.get('x'), ann_vals.get('y')]
        coords = np.array(coords)
        if ann_type == 'polygon' and coords.shape[0] < MIN_POLYGON_POINTS:
            # discard polygon with less than three points
            continue
        # save annoation type, dictionary records, coordinate values
        new_element = {'type': ann_type, 'records': ann_records, 'vals': coords}
        leaf_data.append(new_element)

    # match leaf info (contour and point coordinates) by leaf ordinality
    leaf_info = []
    polygons = ['None'] * MAX_LEAVES_PER_PLANT
    points1 = ['None'] * MAX_LEAVES_PER_PLANT
    points2 = ['None'] * MAX_LEAVES_PER_PLANT

    num_data = len(leaf_data)
    for i in range(num_data):
        curr_type = leaf_data[i].get('type')
        curr_record = leaf_data[i].get('records')
        curr_vals = leaf_data[i].get('vals')
        groupID = -1

        # get groupID  - is this 1st leaf, 2nd leaf ,3rd leaf, etc.
        for j in range(MAX_LEAVES_PER_PLANT):
            g_ID = DIC_GROUP_IDS[j]
            if g_ID in curr_record:
                if g_ID in range(1967,1977):
                    # 1st to 10th leaf (according to dictionary records)
                    groupID = g_ID - 1966
                elif g_ID in range(2044,2064):
                    # 11th to 30th leaf (according to dictionary records)
                    groupID = g_ID - 2033
                break

        if groupID > 0:
            if curr_type == 'point':
                if DIC_POSITION_START in curr_record or DIC_POSITION_LOWER in curr_record or DIC_POSITION_BASAL in curr_record:
                    # this is the start point of leaf
                    points1[groupID - 1] = curr_vals
                elif DIC_POSITION_END in curr_record or DIC_POSITION_UPPER in curr_record:
                    # this is the end point of leaf
                    points2[groupID - 1] = curr_vals
            elif curr_type == 'polygon':
                # this is the polygon of a leaf contour
                polygons[groupID - 1] = curr_vals

    for i in range(MAX_LEAVES_PER_PLANT):
        # save leaf info if it contains polygon and two extreme points
        if not(type(points1[i]) is str) and not(type(points2[i]) is str) and not(type(polygons[i]) is str):
            new_element = {'polygon': polygons[i], 'p1': points1[i], 'p2': points2[i], 'leafID': (i+1)}
            leaf_info.append(new_element)

    return leaf_info


def poly2mask(vertex_row_coords, vertex_col_coords, shape):
    fill_row_coords, fill_col_coords = skimage.draw.polygon(vertex_row_coords, vertex_col_coords, shape)
    mask = np.zeros(shape, dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask


def create_aligned_leaves(out_dir, img_path, leaf_info, task_ids):

    I0 = skimage.io.imread(img_path)

    if TASK_ID_172 in task_ids: # ... or 173 or 174
        # note: images in ann_task_172 are rotated 180 degrees
        I0 = skimage.transform.rotate(I0, 180, resize=False)

    h, w, _ = I0.shape
    num_leaves = len(leaf_info)
    for i in range(num_leaves):
        # get polygon and extreme points
        p = leaf_info[i].get('polygon')
        p1 = leaf_info[i].get('p1')
        p2 = leaf_info[i].get('p2')
        leafID = leaf_info[i].get('leafID')

        # find min/max polygon coordinates
        minX = min(p[:, 0])
        minX = max(0, minX)
        maxX = max(p[:, 0])
        maxX = min(maxX,w-1)
        minY = min(p[:, 1])
        minY = max(0, minY)
        maxY = max(p[:, 1])
        maxY = min(maxY, h-1)

        # make sure min/max X/Y are integers
        minX = int(minX)
        maxX = int(maxX)
        minY = int(minY)
        maxY = int(maxY)

        # extract leaf mask by polygon defining contour
        I1 = I0[minY:maxY+1, minX:maxX+1, :].copy()
        BW = poly2mask(p[:, 1], p[:, 0], (h, w))
        BW = BW[minY:maxY+1, minX:maxX+1]
        if DO_ERODE == 1:
            BW = skimage.morphology.erosion(BW, se)

        # black-out pixels outside leaf mask
        I1[:, :, 0] = np.where(BW, I1[:, :, 0], 0)
        I1[:, :, 1] = np.where(BW, I1[:, :, 1], 0)
        I1[:, :, 2] = np.where(BW, I1[:, :, 2], 0)
        I1 = np.asarray(I1)

        # create alpha channel
        I1_Alpha = np.ones((I1.shape[0], I1.shape[1], 1))
        I1_Alpha[:, :, 0] = np.clip(np.where(BW, I1_Alpha[:, :, 0], 0), 0, 1)

        # apply rotation for alignment by 2 extreme points
        angle = np.arctan2(float(p2[0]) - float(p1[0]), -(float(p2[1]) - float(p1[1])))
        I1 = skimage.transform.rotate(I1, math.degrees(angle), resize=True)
        I1_Alpha = skimage.transform.rotate(I1_Alpha, math.degrees(angle), resize=True)

        # resize, if needed
        if DO_RESIZE == 1:
            I1 = skimage.transform.resize(I1, (I1.shape[0] * RESIZE_SCALE, I1.shape[1] * RESIZE_SCALE))
            I1_Alpha = skimage.transform.resize(I1_Alpha, (I1_Alpha.shape[0] * RESIZE_SCALE, I1_Alpha.shape[1] * RESIZE_SCALE))
        if DO_FIXED_OUTSIZE == 1:
            s = (FIXED_OUTSIZE, FIXED_OUTSIZE)
            I1 = skimage.transform.resize(I1, s)
            I1_Alpha = skimage.transform.resize(I1_Alpha, s)
            I1_Alpha = np.clip(I1_Alpha, 0, 1)

        # extract out_file name
        tmp_str = img_path.split('/')[-1]
        tmp_str = tmp_str.split('.')[-2]
        tmp_str = tmp_str + '_leaf_' + '{:02d}'.format(leafID) + '.png'
        out_file = out_dir + tmp_str

        # wrte out_file
        if DO_ALPHA_LEAVES == 0:
            skimage.io.imsave(out_file, skimage.img_as_ubyte(I1))
        else:
            skimage.io.imsave(out_file, skimage.img_as_ubyte(np.concatenate((I1, I1_Alpha), axis=2)))

    return


def main():

    # for annotation task_ID_XXX, use appropriate values of TASK_ID_XXX , IMG_DIR_XXX , JSON_DIR_XXX , OUT_DIR_XXX
    # task_ids = [TASK_ID_170]
    # json_dir = JSON_DIR_170
    # img_dir = IMG_DIR_170
    # out_dir = OUT_DIR_170

    # task_ids = [TASK_ID_171]
    # json_dir = JSON_DIR_171
    # img_dir = IMG_DIR_171
    # out_dir = OUT_DIR_171

    task_ids = [151,169,170,171,184,185,186]
    json_dir = JSON_DIR_AVOCADO_ABCD
    img_dir = IMG_DIR_AVOCADO_ABCD
    out_dir = OUT_DIR_AVOCADO_ABCD

    # task_ids = [TASK_ID_172]
    # json_dir = JSON_DIR_172
    # img_dir = IMG_DIR_172
    # out_dir = OUT_DIR_172

    # task_ids = [TASK_ID_103, TASK_ID_105, TASK_ID_107]
    # json_dir = JSON_DIR_103
    # img_dir = IMG_DIR_103
    # out_dir = OUT_DIR_103

    # get json and image file names
    json_files = [file_name for file_name in os.listdir(json_dir) if file_name.endswith('.json')]
    num_files = len(json_files)
    img_paths = ['None'] * num_files
    json_paths = ['None'] * num_files
    for i in range(num_files):
        json_paths[i] = json_dir + json_files[i]
        tmp_str = json_files[i].split('_annnotations.json')[-2]
        for j in range(len(IMG_FILE_EXTENSIONS)):
            img_file = img_dir + tmp_str + IMG_FILE_EXTENSIONS[j]
            if os.path.exists(img_file):
                img_paths[i] = img_file
                break

    # get annotations
    print('START PARSING JSON FILES')
    leaf_annotations = []
    for i in range(num_files):
        json_path = json_paths[i]
        print(str(i) + ' : ' + json_path)
        with open(json_path, 'r') as f:
            json_data = json.load(f)
            leaf_info = parse_json_task_171(json_data, task_ids)
            curr = {'img_path': img_paths[i], 'json_path': json_path, 'info': leaf_info}
            leaf_annotations.append(curr)
    print('END PARSING JSON FILES')

    # create images of aligned leaves
    print('START WRITING LEAF IMAGES')
    for i in range(num_files):
        img_path = leaf_annotations[i].get('img_path')
        leaf_info = leaf_annotations[i].get('info')
        if os.path.exists(img_path) and len(leaf_info) > 0:
            print(str(i) + ' : ' + img_path)
            create_aligned_leaves(out_dir, img_path, leaf_info, task_ids)
    print('END WRITING LEAF IMAGES')
    dummy = 0


if __name__ == "__main__":
   main()