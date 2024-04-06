
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
        folder = "specify which A folder to use for training"
        leaf_type = "specify which leaf type to use for training"
        image_size = "specify the image size to use for training" 

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

    class InfoReference:
        description = "Prints information about the model saved in the model-info variable"
        model_path = "Path to a .h5 trained model file"
