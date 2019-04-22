# coding: utf-8

# # Mask R-CNN demo
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

#config_file = "../configs/e2e_mask_rcnn_R_50_FPN_1x.yaml"
config_file = "../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

# Now we create the `COCODemo` object. It contains a few extra options for conveniency, such as the confidence threshold for detections to be shown.

def detect_person(cfg,image):
    coco_demo = COCODemo(
        cfg,
        min_image_size=800,
        confidence_threshold=0.7,
    )
    predictions = coco_demo.compute_prediction(image)
    top_predictions = coco_demo.select_top_predictions(predictions)

    #result = coco_demo.overlay_class_names(result, top_predictions)

    labels = top_predictions.get_field("labels").tolist()
    labels = [coco_demo.CATEGORIES[i] for i in labels]

    if 'person' in labels:
        return 1
    else:
        return 0




# Let's define a few helper functions for loading images

def load(path):
    """
    Given path an image, downloads the image and
    returns a PIL image
    """
    pil_image = Image.open(path).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image

def imshow(img):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")


# Let's now load an image from the COCO dataset. It's reference is in the comment
file_path = "/data/data_67L46I1z/Mask-RCNN-Pedestrian-Detection/INRIAPerson/Train/pos"
#image_path = "/data/data_67L46I1z/maskrcnn-benchmark/demo/3915380994_2e611b1779_z.jpg"

files = os.listdir(file_path)

for index in range(len(files)):
    image_path = os.path.join(file_path,files[index])
    image = load(image_path)
    #imshow(image)

    # ### Computing the predictions
    # We provide a `run_on_opencv_image` function, which takes an image as it was loaded by OpenCV (in `BGR` format), and computes the predictions on them, returning an image with the predictions overlayed on the image.
    # compute predictions
    predictions = detect_person(cfg,image)
    print(bool(predictions))
