from Mask_RCNN.mrcnn.config import Config
import os
import sys
SET_TYPE="general"
# SET_TYPE="ortho"
ROOT_DIR = os.getcwd()
sys.path.append(ROOT_DIR+"/Mask_RCNN")  # To find local version of the library

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))#
MODEL_DIR = os.path.join(ROOT_DIR,"models/")
GENERAL_MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_general-set_0050.h5")
GENERAL_IMAGE_DIR = os.path.join("data", "General_Surgery_Set/test")
ORTHO_MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_general-set_0050.h5")
ORTHO_IMAGE_DIR = os.path.join("data", "orthopaedic/test")
ORTHO_MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_orthopaedic_0120.h5")


MODEL_PATH_CATEGORY = os.path.join(MODEL_DIR,"sub_category_mask_RCNN_weights/mask_rcnn_supercat_0080.h5")
MODEL_PATH_SCISSORS = os.path.join(MODEL_DIR,"sub_category_mask_RCNN_weights/mask_rcnn_scissor_0120.h5")
MODEL_PATH_RING_FORCEPS = os.path.join(MODEL_DIR,"webcam_ringforceps_0120.h5")#os.path.join(MODEL_DIR,"sub_category_mask_RCNN_weights/mask_rcnn_ringforceps_0120.h5")
MODEL_PATH_THUMB_FORCEPS = os.path.join(MODEL_DIR,"sub_category_mask_RCNN_weights/mask_rcnn_thumbforcepsdissectors_0120.h5")
MODEL_PATH_SCALPEL_BP_HANDLE = os.path.join(MODEL_DIR,"sub_category_mask_RCNN_weights/mask_rcnn_scalpelbphandles_0095.h5")
MODEL_PATH_ORTHO = os.path.join(MODEL_DIR, "mask_rcnn_orthopaedic_0120.h5")


class FlawSetConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "flawing"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
        
    NUM_CLASSES = 3  # Background + 18 classes (6_Babcock_Tissue_Forceps, 6_Mayo_Needle_Holder, 7_Metzenbaum_Scissors, 8_Babcock_Tissue_Forceps, 8_Mayo_Needle_Holder, 9_DeBakey_Dissector, 9_Metzenbaum_Scissors, Allis_Tissue_Forceps, Bonneys_Non_Toothed_Dissector, Bonneys_Toothed_Dissector, Curved_Mayo_Scissors, Dressing_Scissors, Gillies_Toothed_Dissector, Lahey_Forceps, No3_BP_Handle, No4_BP_Handle, Sponge_Forceps, Crile_Artery_Forceps)
   
    # Use smaller images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 480#512
    IMAGE_MAX_DIM = 640#768

    # Number of training steps per epoch
    #STEPS_PER_EPOCH = 300

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.5

    RPN_NMS_THRESHOLD = 0.9
    
    VALIDATION_STEPS = 100


