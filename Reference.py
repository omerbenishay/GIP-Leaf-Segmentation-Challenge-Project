
class HelpReference:

    description = "\
    Model-leaf uses the Tensorflow implementation of Mask-RCNN by MatterPort and a handful of integration scripts and utilities to simplify training and inference of leaf datasets.\
    For information on the different subcommands read the according manual pages\
    "

    help_description = "\
    Prints the synopsis and the list of possible options and commands."

    class TrainReference:
        description = "Creates a dataset of synthetic pictures, and runs the training model on the dataset. The best result model is saved as a .h5 file."
        output = "specify path to .h5 model location [default: current]"
        dataset_keep = "specify how many samples to keep [default: 0]"
        test_set = "specify path to test set"
        config = "specify path to the model (mask-r cnn) config file"
        synthetic = "Set the synthetic dataset generator to scatter the leaves randomly (cucumber), or to group the leaves around a base (banana)"
        leaf_size_min = "Set the minimum size of leaves in the synthetic picture"
        leaf_size_max = "Set the maximum size of leaves in the synthetic picture"
        preview_only = "generate samples of training set without training the model"
        dataset_class = "dataset module and class name to use [eg: 'BananaDataset']"
        dataset_config = "dataset configuration file path"
        epochs = "number of training epochs"
        steps_per_epoch = "number of training steps to perform per epoch"
        layers = "layers of model to train. Other layers will remain unchanged"
        pretrain = "path to a .h5 file with a pretrained model, or just 'COCO' to retrieve\
        the coco pretrain file. [default: COCO]" 

    class InferReference:
        description = "Loads a dataset, loads a model, runs inference on all the pictures located in a directory. Outputs a set of pictures with a translucent mask on every detected leaf. Additionally, a json annotation file is generated."
        output = "Set output directory [default: current]"
        no_pictures = "Create only infered pictures with colorful transparent masks"
        no_contours = "Create contour annotation file only"
        path = "path to directory containing images to infer or path to image to infer"
        model = "path to .h5 trained model to infer with"
        no_masks = "do not save mask images"
        task = "task id for agrinet datasets"
        gt = "Dataset adapter name"

    class CutReference:
        description = "Cut single leaf pictures from an annotated dataset"
        normalize = "Normalize the generated pictures to the specified width-size. By default pictures are not normalized, every leaf stays at its original size"
        background = "Specify the background of the leaf, transparent means keep the alpha channel [default: transparent]"
        limit = "Maximum number of object files to create. Not specifying this argument will result in cutting all the available objects"
        path = "path to json file in COCO format, with relative image paths"
        output = "Set output directory [default: current]"
        adapter = "Type of annotation - specify in order to correctly parse the annotation file"
        rotate = "Rotate output files to match 2 points from annotation"
        task = "task id for agrinet datasets"

    class InfoReference:
        description = "Prints information about the model saved in the model-info variable"
        model_path = "Path to a .h5 trained model file"

    class DownloadReference:
        description = "Download specific task id to local directory"
        task_id = "Task id number e.g: 103"
        location = "Location to save the files"
