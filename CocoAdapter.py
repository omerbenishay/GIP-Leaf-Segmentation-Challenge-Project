import json
import os
from BaseAnnotationAdapter import BaseAnnotationAdapter


class CocoAdapter(BaseAnnotationAdapter):
    def __init__(self, coco_file_path, n=None):
        with open(coco_file_path, 'r') as f:
            self.data = json.load(f)
        self.n = n
        self.category = 'leaf'
        self.dir_path = os.path.dirname(coco_file_path)
        self.generator = self._cut_jobs()

    def _cut_jobs(self):
        annotations = self._annotation_generator()
        for i, leaf_annotation in enumerate(annotations):
            image_relative_path = self._get_image_path(leaf_annotation["image_id"])
            image_path = os.path.join(self.dir_path, image_relative_path)
            yield (leaf_annotation.get("segmentation")[0], image_path, i)

    def _annotation_generator(self):
        category_id = [category["id"] for category in self.data["categories"] if category["name"] == self.category][0]
        if category_id is None:
            return None
        count = 0
        for annotation in self.data["annotations"]:
            if self.n is not None and count == self.n:
                break
            # Discard objects with more than one polygon
            if len(annotation.get("segmentation")) != 1:
                continue
            # Discard objects of size 0
            if annotation["bbox"][2] < 1 or annotation["bbox"][3] < 1:
                continue
            if annotation.get("category_id") == category_id:
                count += 1
                yield annotation

    def _get_image_path(self, image_id):
        images = self.data.get("images")
        image_path = [image["file_name"] for image in images if image["id"] == image_id][0]
        return image_path

    def __next__(self):
        """
        Needs to be implemented in order to iterate through the annotations

        :return:    A tuple (leaf_annotation, image_path, i) where
                    leaf annotation is a list of coordinates [x,y,x,y,x,y,...]
                    image_path is a full or relative path to the image
                    i is a running index used for example to name the leaf picture
        """
        return next(self.generator)

