from tqdm import tqdm
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import detectron2
import glob
import os
import cv2
import numpy as np


image_path = './Group_Component/sequence/'
proc_images = sorted(glob.glob(os.path.join(image_path, '*.jpg')))

# font
font = cv2.FONT_HERSHEY_SIMPLEX

# org
org = (50, 50)

# fontScale
fontScale = 0.8

# Blue color in BGR
color = (255, 0, 0)

# Line thickness of 2 px
thickness = 2

drawing = False
start_image = cv2.imread(proc_images[0])
cv2.putText(start_image, "Draw a rectangle and press Enter When you're ready",
            org, font, fontScale, color, thickness, cv2.LINE_AA)
display = start_image.copy()
x1 = y1 = x2 = y2 = -1


def draw_rectangle(event, x, y, flags, param):
    global x1, y1, x2, y2, drawing, display

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        display = start_image.copy()
        x1, y1 = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            display = start_image.copy()
            cv2.rectangle(display, (x1, y1), (x, y), (0, 0, 255), thickness=3)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x2, y2 = x, y
        cv2.rectangle(display, (x1, y1), (x, y), (0, 0, 255), thickness=3)


cv2.namedWindow('Task2')
cv2.setMouseCallback('Task2', draw_rectangle)

while(1):
    cv2.imshow('Task2', display)
    if (cv2.waitKey(20) & 0xFF == 13):
        if (x1 == -1 or x2 == -1 or y1 == -1 or y2 == -1):
            print('Please draw a rectangle')
            continue
        if (abs(x1-x2) < 100 or abs(y1-y2) < 100):
            print('Your rectangle too small')
            continue
        break

cv2.waitKey(int(1000/60))
setup_logger()

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
# WITHOUT CUDA
cfg.MODEL.DEVICE = 'cpu'

predictor = DefaultPredictor(cfg)


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def checkOverlap(l1, r1, l2, r2):
    # If one rectangle is on left side of other:
    if l1.x > r2.x or l2.x > r1.x:
        return False
    # If one rectangle is above left side of other:
    if l1.y > r2.y or l2.y > r1.y:
        return False
    return True


for image_path in (proc_images):
    n_in = n_out = 0
    display = cv2.imread(image_path)
    outputs = predictor(display)
    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), thickness=3)

    for i, box in enumerate(outputs["instances"].pred_boxes):
        if outputs["instances"].pred_classes[i] == 0:
            x_1, y_1, x_2, y_2 = [int(i) for i in box.tolist()]
            cv2.rectangle(display, (x_1, y_1), (x_2, y_2),
                          (0, 255, 0), thickness=2)
            if checkOverlap(Point(x1, y1), Point(x2, y2), Point(x_1, y_1), Point(x_2, y_2)):
                n_in += 1
            else:
                n_out += 1

    print('In: %d\nOut: %d' % (n_in, n_out))

    cv2.putText(display, "In: %d" % (n_in),
                (30, 30), font, fontScale, (0, 0, 0), thickness, cv2.LINE_AA)

    cv2.putText(display, "Out: %d" % (n_out),
                (30, 60), font, fontScale, (0, 0, 0), thickness, cv2.LINE_AA)

    cv2.imshow('Task2', display)
    cv2.waitKey(int(1000/60))

    # break

cv2.destroyAllWindows()
