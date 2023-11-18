import numpy as np
import json
import os
from BaseAnnotationAdapter import BaseAnnotationAdapter


class AgrinetAdapter(BaseAnnotationAdapter):
    IMAGE_FORMATS = ['jpg', 'jpeg', 'png', 'bmp']

    def __init__(self, annotation_path, task_id, n=None):

        self.task_id = int(task_id)

        annotation_file_paths = [annotation_path]
        self.dir_path = os.path.dirname(annotation_path)

        if os.path.isdir(annotation_path):
            annotation_file_paths = [os.path.join(annotation_path, file_name) for file_name in
                                     os.listdir(annotation_path) if file_name.endswith(".json")]
            self.dir_path = annotation_path

        self.data = {}
        for file_path in annotation_file_paths:
            with open(file_path, 'r') as f:
                self.data.update(json.load(f))
        self.n = n
        self.annotations = {}
        try:
            self._annotation_generate()
        except Exception as e:
            print("Bad annotation format")

        self.generator = self._cut_jobs()

    def _cut_jobs(self):
        # polygon_items = [annotation for annotation in self.annotations.items() if annotation[]]
        for i, (leaf_key, leaf_data) in enumerate(self.annotations.items()):
            image_relative_path = self._get_image_path(leaf_data["image"])
            if image_relative_path is None:
                print("Image for {} does not exist".format(leaf_data["image"]))
                continue
            image_path = os.path.join(self.dir_path, image_relative_path)
            # Save reference of index to original annotation to retrieve 'point' if needed
            yield leaf_data["polygon"], image_path, i

    def _annotation_generate(self):
        for image_key, annotations_array in self.data.items():

            for annotation_data in annotations_array:
                _id = annotation_data["_id"]
                is_valid_record = (annotation_data["deleted"] == 0 and
                                   annotation_data["annotation_task_id"] == self.task_id)

                if is_valid_record and annotation_data["annotation_type"] == "polygon":
                    self.annotations[_id] = {}
                    # Flatten the coordinates of the polygon to one array
                    leaf_polygon_coords = [[ann['x'], ann['y']] for ann in annotation_data["annotations"]]
                    leaf_polygon = np.array(leaf_polygon_coords).reshape((-1,)).tolist()
                    self.annotations[_id]["polygon"] = leaf_polygon
                    self.annotations[_id]["image"] = annotation_data["frame_id"]

    def _get_image_path(self, image_id):
        file_list = os.listdir(self.dir_path)
        match = [file_name for file_name in file_list if file_name.startswith(str(image_id))
                 and file_name.split(".")[-1].lower() in self.IMAGE_FORMATS]

        if len(match) != 0:
            return match[0]

        return None

    def __next__(self):
        """
        Needs to be implemented in order to iterate through the annotations

        :return:    A tuple (leaf_annotation, image_path, i) where
                    leaf annotation is a list of coordinates [x,y,x,y,x,y,...]
                    image_path is a full or relative path to the image
                    i is a running index used for example to name the leaf picture
        """
        return next(self.generator)
