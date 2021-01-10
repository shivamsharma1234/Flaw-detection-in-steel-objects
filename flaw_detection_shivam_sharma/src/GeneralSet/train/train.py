#%%
import os 
import sys
# Root directory of the project
ROOT_DIR = os.path.dirname(os.path.realpath(__file__))#os.path.abspath(os.getcwd())
import sys
sys.path.append(os.path.join(ROOT_DIR,'../../'))
print("Appended, /src/Mask_RCNN" + " to PATH") #src/ folder contains common files required by all experiments

sys.path.append(os.path.join(ROOT_DIR,'../..//Mask_RCNN'))
print(os.path.join(ROOT_DIR,'../../'))

import json
import datetime
import numpy as np
import skimage.draw
from imgaug import augmenters as iaa
import config
#import paths
# Import Mask RCNN
#from Mask_RCNN.mrcnn.config import Config
from mrcnn import model as modellib, utils
import datasets
import imgaug
import yaml
import argparse
import keras

#pip3 install numpy
#pip3 install scikit-image
#pip3 install imgaug
# Root directory of the project
# ROOT_DIR =  os.getcwd() + "/src/Mask_RCNN"
# Path to trained weights file
# COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
#%%
# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
# DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
set_config = None
args = None

def yml_read(config_yml):

    global set_config
    with open(config_yml) as f:
        set_config = yaml.load(f, Loader=yaml.FullLoader)[args.speciality]
 
def args_parse():

     # Parse command line arguments
    parser = argparse.ArgumentParser(
    description='Train Mask R-CNN. Usage python train.py general --weights=modelfile --dataset ')
    
    parser.add_argument("set_type",
                        metavar="<set_type>",
                        help="'general' or 'orthopaedic'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/custom/dataset/",
                        help='Directory of the custom dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--speciality', required=True,
                        metavar="GeneralSet",
                        default="GeneralSet",
                        help="enter speciality")
    parser.add_argument('--config_yml', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to config yml file")
    parser.add_argument("--reinitialize", required=False, action='store_true')
    parser.add_argument('--logs', required=False,
                        default=os.path.join(ROOT_DIR,"../../../logs"),
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--prev_augment', default=False, 
                    action='store_true')
    global args
    args = parser.parse_args()

def assign(config):
   
    # Configurations
    if args.set_type == "types":
        config = config.FlawSetConfig()
   
    
    #if args.set_type != "ring_forceps":
    # for k, v in set_config['params'][args.set_type].items():
    #    setattr(config, k, v)
    #    print(k,v)
    # print(config)
    config.display()
    return config

def augment():
    if not args.prev_augment:
        # Augmentating images based on Matterport's recommendations: See: https://github.com/matterport/Mask_RCNN/blob/master/samples/nucleus/nucleus.py
        print("Augmenting Dataset")
        augmentation = iaa.Sequential([
            iaa.OneOf([
                        iaa.Fliplr(0.9),
                        iaa.Flipud(0.9)
                      ]),
            iaa.OneOf([ 
                        iaa.Affine(rotate=(-90, 90)),
                        iaa.Affine(rotate=(-30, 30)),
                        iaa.Affine(rotate=(-30, 30)),
                        iaa.Affine(rotate=(-270, 270))
                      ]),
            iaa.Affine(scale=(0.4, 0.6)),
        ])
    return augmentation

def create_model(config_t):
    # Create model
    model = modellib.MaskRCNN(mode="training", config=config_t,
                                  model_dir=args.logs)
    
    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = os.path.join(ROOT_DIR,"../../../file.h5")
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()[1]
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights


    
    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco" :
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        if args.reinitialize:
            model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
        else:
            model.load_weights(weights_path, by_name=True)
    
    return model

def dataset_load():
    # Dataset.
    if args.set_type == "types":
        dataset_train = datasets.FlawDataSet()
        dataset_val = datasets.FlawDataSet()
    
    

    dataset_train.load_custom(args.dataset, "train")
    dataset_train.prepare()
    dataset_val.load_custom(args.dataset, "val")
    dataset_val.prepare()

    return dataset_train,dataset_val

if __name__ == '__main__':
    args_parse()
    yml_read(args.config_yml)
    # Validate arguments
    assert args.dataset, "Argument --dataset is required for training"

    print("Set Type: ", args.set_type)
    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)
    #config = Config()
    config_t = assign(config)

    print('config',config_t.NUM_CLASSES)
    augmentation = augment()

    model = create_model(config_t)

    dataset_train,dataset_val = dataset_load()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    # Training Schedule for Mask RCNN
    print("Train network heads")
    model.train(dataset_train, dataset_val,
            learning_rate=config_t.LEARNING_RATE, # made 10 times
            epochs=40,
            augmentation=augmentation,
            layers='heads')
    
   
    #Finetune layers from ResNet stage 4 and up
    print("Fine tune Resnet stage 4 and up")
    model.train(dataset_train, dataset_val,
            learning_rate=config_t.LEARNING_RATE, # made 10 times
            epochs=120,
            #augmentation=augmentation,
            layers='4+')

    print("Train all layers")
    model.train(dataset_train, dataset_val,
            learning_rate=config_t.LEARNING_RATE/10,  
            epochs=300,
            augmentation=augmentation,
            layers='all')

