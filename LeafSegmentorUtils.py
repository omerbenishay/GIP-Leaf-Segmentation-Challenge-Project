import os
from questionary import select
import h5py


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


def add_metadata_dict_to_h5(filepath, dict_name, dict_content):
    """
    Supports only simple dict of key: value where the values are
    :param filepath:
    :param dict_name:
    :param dict_content:
    :return:
    """
    with h5py.File(filepath, 'a') as f:
        try:
            f.create_group(dict_name)
        except ValueError as e:
            print("dictionary already exists")

        group_attrs = f[dict_name].attrs
        for key, value in dict_content.items():
            if type(value) == str:
                value = str.encode(value)
            try:
                group_attrs[key] = value
            except TypeError as e:
                print("Impossible to insert this dictionary as metadata.\n"
                      "Type {} is not supported as a value".format(type(value)))
                print("Error: {}".format(e))
                return None
        f.flush()


def get_metadata_dict_from_h5(filepath, dict_name):
    """
    Assume byte strings are strings
    :param filepath:
    :param dict_name:
    :return: a dictionary
    """
    metadata_dict = {}
    with h5py.File(filepath, 'r') as f:
        try:
            group_attrs = f[dict_name].attrs
        except ValueError as e:
            print("dictionary of name {} doesn't exist".format(dict_name))
            print("Error: {}".format(e))
            return None
        for key, value in list(group_attrs.items()):
            if type(value) == bytes:
                value = value.decode('utf-8')
            if hasattr(value, '__len__'):
                metadata_dict[key] = value[0] if len(value) == 1 else value
            else:
                metadata_dict[key] = value
    return metadata_dict


def get_clean_dict_from_class(my_class_name):
    class_dict = my_class_name.__dict__
    my_dict = {key: val for key, val in class_dict.items() if not key.startswith('__') and not callable(val)}
    return my_dict
