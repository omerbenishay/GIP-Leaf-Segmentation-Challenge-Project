#from LeafSegmentorDownload import download
from LeafSegmentorTrain import train
from LeafSegmentorInfer import infer
# from LeafSegmentorInfer_corn_Adva_Shani import infer
from Reference import HelpReference
from LeafSegmentorInfo import info
import argparse


def main():
    # top level parser
    parser = argparse.ArgumentParser(description=HelpReference.description, add_help=False)
    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help=HelpReference.help_description)
    parser.add_argument('-v', '--version', action='version', version='%(prog)s 1.0')
    subparsers = parser.add_subparsers()

    # parser for train
    parser_train = subparsers.add_parser('train', help=HelpReference.TrainReference.description)
    parser_train.set_defaults(func=train)
    parser_train.add_argument('-o', '--output', help=HelpReference.TrainReference.output, default='models')
    parser_train.add_argument('-k', '--dataset-keep', type=int, help=HelpReference.TrainReference.dataset_keep,
                              default=0)
    parser_train.add_argument('--preview-only', help=HelpReference.TrainReference.preview_only, action='store_true')
    parser_train.add_argument('-e', '--epochs', type=int, help=HelpReference.TrainReference.epochs, default=10)
    parser_train.add_argument('-s', '--steps-per-epoch', type=int, help=HelpReference.TrainReference.steps_per_epoch,
                              default=0)
    parser_train.add_argument('-l', '--layers', choices=['all', 'heads', '3+', '4+', '5+'],
                              help=HelpReference.TrainReference.layers, default='all')
    parser_train.add_argument('-p', '--pretrain', help=HelpReference.TrainReference.pretrain, default="COCO")
    parser_train.add_argument('dataset_config_file', metavar='dataset-config-file',
                              help=HelpReference.TrainReference.dataset_config)
    parser_train.add_argument('-A', "--Afolder", help=HelpReference.TrainReference.folder, default="1")    
    parser_train.add_argument('-lt', "--leaf-type", choices=['clean', 'clean_smooth', 'clean_smooth_r05', 'org', 'alpha_0.15'], help=HelpReference.TrainReference.leaf_type, default="clean")                       
    parser_train.add_argument('-S', "--image-size", help=HelpReference.TrainReference.image_size, default="512")

    # parser for infer
    parser_infer = subparsers.add_parser('infer', help=HelpReference.InferReference.description)
    parser_infer.set_defaults(func=infer)
    parser_infer.add_argument('-m', '--model', help=HelpReference.InferReference.model, default='./')
    parser_infer.add_argument('-o', '--output', help=HelpReference.InferReference.output, default='outputs')
    parser_infer.add_argument('--no-pictures', help=HelpReference.InferReference.no_pictures, action='store_true')
    parser_infer.add_argument('--no-contours', help=HelpReference.InferReference.no_contours, action='store_true')
    parser_infer.add_argument('--no-masks', help=HelpReference.InferReference.no_masks, action='store_true')
    parser_infer.add_argument('--gt', help=HelpReference.InferReference.gt, default=None)
    parser_infer.add_argument('--task', type=int, help=HelpReference.InferReference.task, default=None)
    parser_infer.add_argument('path', help=HelpReference.InferReference.path)

    # parser for info
    parser_info = subparsers.add_parser('info', help=HelpReference.InfoReference.description)
    parser_info.set_defaults(func=info)
    parser_info.add_argument('model_path', help=HelpReference.InfoReference.model_path, default='models')

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
