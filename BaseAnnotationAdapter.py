
class BaseAnnotationAdapter(object):
    def __iter__(self):
        return self

    def __next__(self):
        """
        Needs to be implemented in order to iterate through the annotations

        :return:    A tuple (leaf_annotation, image_path, i) where
                    leaf annotation is a list of coordinates [x,y,x,y,x,y,...]
                    image_path is a full or relative path to the image
                    i is a running index used for example to name the leaf picture
        """
        raise NotImplementedError()


class RotatableAdapter(BaseAnnotationAdapter):
    def __next__(self):
        raise NotImplementedError()

    def get_point(self, index):
        """
        Get the points of a leaf if they exist in the annotation file
        :param index:   Index passed with the leaf annotation by the collection
        :return:        A tuple of arrays ([x, y],[x,y]) with the location of the points if they exist
                        if points doesn't exists for this leaf, returns None
        """
        raise NotImplementedError()
