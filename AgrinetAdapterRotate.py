import numpy as np
import json
import os
from BaseAnnotationAdapter import RotatableAdapter


class AgrinetAdapterRotate(RotatableAdapter):
    IMAGE_FORMATS = ['jpg', 'jpeg', 'png', 'bmp']

    # ID's of relevant dictionary records from Phenomics dictionary
    # dictionary records [1967 .. 1976] correspond to group [1st .. 10th]
    # dictionary records [2044 .. 2063] correspond to group [11th .. 30th]
    DIC_GROUP_IDS = [i for i in range(1967, 1977)] + [i for i in range(2044, 2064)]
    # minimum number of points per polygon
    MIN_POLYGON_POINTS = 3
    # maximum number of leaves per plant
    MAX_LEAVES_PER_PLANT = len(DIC_GROUP_IDS)
    # start/end points of a leaf
    DIC_POSITION_START = 1962
    DIC_POSITION_END = 1963
    DIC_POSITION_UPPER = 1964  # same as DIC_POSITION_END
    DIC_POSITION_LOWER = 1966  # same as DIC_POSITION_START
    DIC_POSITION_BASAL = 1993  # same as DIC_POSITION_START
    #TASK_IDS = [103, 105, 107, 171]
    TASK_IDS = [172, 173, 174]
    POINT1_INDEX = 2
    POINT2_INDEX = 3

    def __init__(self, annotation_path, task_id, n=None):

        self.task_id = int(task_id)
        self.dir_path = annotation_path
        self.n = n

        self.pictures_list = self.list_all_pictures()
        self.annotation_dictionary = self.retrieve_annotation_data()
        self.leaf_annotations = self._parse_all_annotations()
        self.generator = self._generator()

    def get_point(self, index):
        """
        Get the points of a leaf if it exists in the annotation file
        :param index:   Index passed with the leaf annotation by the collection
        :return:        an array [[x, y], [x,y]] with the location of the points if they exists
                        if points doesn't exists, returns None
        """
        point1 = self.leaf_annotations[index][self.POINT1_INDEX]
        point2 = self.leaf_annotations[index][self.POINT2_INDEX]
        return [list(point1), list(point2)]

    def list_all_pictures(self):
        def _is_picture(file_name):
            return file_name.split(".")[-1] in self.IMAGE_FORMATS

        return [os.path.join(self.dir_path, file_name) for file_name in
                os.listdir(self.dir_path) if _is_picture(file_name)]

    def retrieve_annotation_data(self):
        annotation_dict = {}
        annotation_file_paths = [os.path.join(self.dir_path, file_name) for file_name in
                                 os.listdir(self.dir_path) if file_name.endswith(".json")]
        for picture_path in self.pictures_list:
            path_without_extension = os.path.splitext(picture_path)[0]
            try:
                annotation_file_name = next(file for file in annotation_file_paths if path_without_extension in file)
            except StopIteration as e:
                print("Picture in directory without annotation, skipping...")
                continue
            with open(annotation_file_name) as f:
                annotation_dict[picture_path] = json.load(f)

        return annotation_dict

    def _parse_all_annotations(self):
        parsed_annotations = []
        for pict_path, annotation in self.annotation_dictionary.items():
            parsed_annotations.extend(self.parse_json_task(annotation, pict_path))

        return parsed_annotations

    def parse_json_task(self, json_data, image_path):
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
            if not (ann_task_id in self.TASK_IDS):
                # discard this annotation if it doesn't belong to the scecified task Id's
                continue
            ann_type = curr.get('annotation_type')
            if not (ann_type == 'polygon' or ann_type == 'point'):
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
            if ann_type == 'polygon' and coords.shape[0] < self.MIN_POLYGON_POINTS:
                # discard polygon with less than three points
                continue
            # save annoation type, dictionary records, coordinate values
            new_element = {'type': ann_type, 'records': ann_records, 'vals': coords}
            leaf_data.append(new_element)

        # match leaf info (contour and point coordinates) by leaf ordinality
        leaf_info = []
        polygons = ['None'] * self.MAX_LEAVES_PER_PLANT
        points1 = ['None'] * self.MAX_LEAVES_PER_PLANT
        points2 = ['None'] * self.MAX_LEAVES_PER_PLANT

        num_data = len(leaf_data)
        for i in range(num_data):
            curr_type = leaf_data[i].get('type')
            curr_record = leaf_data[i].get('records')
            curr_vals = leaf_data[i].get('vals')
            groupID = -1

            # get groupID  - is this 1st leaf, 2nd leaf ,3rd leaf, etc.
            for j in range(self.MAX_LEAVES_PER_PLANT):
                g_ID = self.DIC_GROUP_IDS[j]
                if g_ID in curr_record:
                    if g_ID in range(1967, 1977):
                        # 1st to 10th leaf (according to dictionary records)
                        groupID = g_ID - 1966
                    elif g_ID in range(2044, 2064):
                        # 11th to 30th leaf (according to dictionary records)
                        groupID = g_ID - 2033
                    break

            if groupID > 0:
                if curr_type == 'point':
                    if self.DIC_POSITION_START in curr_record or self.DIC_POSITION_LOWER in curr_record or self.DIC_POSITION_BASAL in curr_record:
                        # this is the start point of leaf
                        points1[groupID - 1] = curr_vals
                    elif self.DIC_POSITION_END in curr_record or self.DIC_POSITION_UPPER in curr_record:
                        # this is the end point of leaf
                        points2[groupID - 1] = curr_vals
                elif curr_type == 'polygon':
                    # this is the polygon of a leaf contour
                    polygons[groupID - 1] = curr_vals

        for i in range(self.MAX_LEAVES_PER_PLANT):
            # save leaf info if it contains polygon and two extreme points
            if not (type(points1[i]) is str) and not (type(points2[i]) is str) and not (type(polygons[i]) is str):
                # leaf_polygon_coords = [[ann['x'], ann['y']] for ann in polygons[i]]
                leaf_polygon = polygons[i].reshape((-1,)).tolist()
                leaf_tuple = (leaf_polygon, image_path, points1[i], points2[i])
                # new_element = {'polygon': polygons[i], 'p1': points1[i], 'p2': points2[i], 'leafID': (i + 1)}
                leaf_info.append(leaf_tuple)

        return leaf_info

    def _generator(self):
        for i, leaf_annotation in enumerate(self.leaf_annotations):
            yield (*leaf_annotation[:2], i)

    def __next__(self):
        """
        Needs to be implemented in order to iterate through the annotations

        :return:    A tuple (leaf_annotation, image_path, i) where
                    leaf annotation is a list of coordinates [x,y,x,y,x,y,...]
                    image_path is a full or relative path to the image
                    i is a running index used for example to name the leaf picture
        """
        return next(self.generator)
