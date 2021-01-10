import os
import sys
import json
import datetime
import numpy as np
import skimage.draw


# Import Mask RCNN
ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(ROOT_DIR)  # To find local version of the library
#from mrcnn.config import Config
from mrcnn import model as modellib, utils
import yaml



class FlawDataSet(utils.Dataset):

    def load_custom(self, dataset_dir, subset):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes.
        
        train_set = []
        #Read the speciality yml file 
        with open(os.path.join(ROOT_DIR,'speciality.yaml')) as f:
            train_set = yaml.load(f, Loader=yaml.FullLoader)['Flaws']['types']
            

        try:
            train_set.remove('BG')
        except:
            pass
        # Add classes.
        for i in range(len(train_set)):
            print("types", i+1, train_set[i])
            self.add_class("types", i+1, train_set[i])
        
                
        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # We mostly care about the x and y coordinates of each region
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        #print(annotations)
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            polygons_org = [r['shape_attributes'] for r in a['regions']]
            objects = [s['region_attributes'] for s in a['regions']]

            num_ids = []
            polygons = []

            # Component is what I defined the class in VGG Image Annotator
            for n,p in zip(objects, polygons_org):
                try:
                   
                    
                    for i in range(len(train_set)):
                        if n['Component'] == train_set[i]:
                            num_ids.append(i+1)
                            polygons.append(p)
                            break
                            
                            #print(class_names.set_2_2_mm_case[i])
                    
                except:
                    pass

            
            
            # if no detection, skip
            if(len(polygons)==0):
                continue
            
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "types",  # for a single class just add the name here
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                num_ids=num_ids)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "types":
            return super(self.__class__, self).load_mask(image_id)

        info = self.image_info[image_id]
        num_ids = info['num_ids']
        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)

        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Map class names to class IDs.
        num_ids = np.array(num_ids, dtype=np.int32)
        return mask, num_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "types":
            return info["types"]
        else:
            super(self.__class__, self).image_reference(image_id)


