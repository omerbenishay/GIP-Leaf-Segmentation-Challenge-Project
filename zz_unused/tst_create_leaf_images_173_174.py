
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

# maize plantlets - annotation tasks 173,174 ONLY !

TASK_ID_173 = 173
IMG_DIR_173 = 'C:/alon/Phenomics_tst_images_jsons/task_173_bmp/'
JSON_DIR_173 = 'C:/alon/Phenomics_tst_images_jsons/task_173_bmp/'
OUT_DIR_173 = 'C:/alon/agriculture/corn_ann_173/tst_leaves_180/'

TASK_ID_174 = 174
IMG_DIR_174 = 'C:/alon/Phenomics_tst_images_jsons/task_174_bmp/'
JSON_DIR_174 = 'C:/alon/Phenomics_tst_images_jsons/task_174_bmp/'
OUT_DIR_174 = 'C:/alon/agriculture/corn_ann_174/tst_leaves_aligned_bmp/'

TASK_ID_213 = 213
IMG_DIR_213 = 'C:/alon/Phenomics_tst_images_jsons/task_213/'
JSON_DIR_213 = 'C:/alon/Phenomics_tst_images_jsons/task_213/'
OUT_DIR_213 = 'C:/alon/agriculture/corn_ann_213/tst_leaves_1/'

TASK_ID_214 = 214
IMG_DIR_214 = 'C:/alon/Phenomics_tst_images_jsons/task_214/'
JSON_DIR_214 = 'C:/alon/Phenomics_tst_images_jsons/task_214/'
OUT_DIR_214 = 'C:/alon/agriculture/corn_ann_214/tst_leaves_1/'

TASK_ID_215 = 215
IMG_DIR_215 = 'C:/alon/Phenomics_tst_images_jsons/task_215/'
JSON_DIR_215 = 'C:/alon/Phenomics_tst_images_jsons/task_215/'
OUT_DIR_215 = 'C:/alon/agriculture/corn_ann_215/tst_leaves_1/'

# valid image file extensions
IMG_FILE_EXTENSIONS = ['.bmp', '.jpg', '.jpeg', '.png', '.tiff']


# minimum number of points per polygon
MIN_POLYGON_POINTS = 3
# maximum number of leaves per plant
MAX_LEAVES_PER_PLANT = 50

# parameters for output leaf images
DO_ERODE = 0            # specify for small erosion of leaf + mask
se = skimage.morphology.disk(radius=5)     # erosion kernel
DO_RESIZE = 0           # specify for leaf resize
RESIZE_SCALE = 0.15     # resize scale
DO_FIXED_OUTSIZE = 0    # specify for fixing max(out_image_height, out_image_width)
FIXED_OUTSIZE = 128     # 128  256  512
DO_ALPHA_LEAVES = 1     # write alpha channel in output file


def parse_json_task_173_174(json_data, task_ids):
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

        leaf_id = i//3
        point_id = i%3
        if curr_type == 'polygon':
            polygons[leaf_id] = curr_vals
        elif curr_type == 'point':
            if point_id == 1:
                points2[leaf_id] = curr_vals
            elif point_id == 2:
                points1[leaf_id] = curr_vals

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
    if TASK_ID_173 in task_ids:
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
        #angle = 0.0
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

    # task_ids = [TASK_ID_173]
    # json_dir = JSON_DIR_173
    # img_dir = IMG_DIR_173
    # out_dir = OUT_DIR_173

    # task_ids = [TASK_ID_174]
    # json_dir = JSON_DIR_174
    # img_dir = IMG_DIR_174
    # out_dir = OUT_DIR_174

    # task_ids = [TASK_ID_213]
    # json_dir = JSON_DIR_213
    # img_dir = IMG_DIR_213
    # out_dir = OUT_DIR_213

    # task_ids = [TASK_ID_214]
    # json_dir = JSON_DIR_214
    # img_dir = IMG_DIR_214
    # out_dir = OUT_DIR_214

    task_ids = [TASK_ID_215]
    json_dir = JSON_DIR_215
    img_dir = IMG_DIR_215
    out_dir = OUT_DIR_215

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
            leaf_info = parse_json_task_173_174(json_data, task_ids)
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