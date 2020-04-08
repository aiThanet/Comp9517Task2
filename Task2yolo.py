from tqdm import tqdm
import glob
import os
import cv2
import numpy as np
import sys
from sys import platform

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *


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
running_state = False
start_image = cv2.imread(proc_images[0])
cv2.putText(start_image, "Draw a rectangle and press Enter When you're ready",
            org, font, fontScale, color, thickness, cv2.LINE_AA)
display = start_image.copy()
x1 = y1 = x2 = y2 = -1



def initialize_model(cfg='cfg/yolov3-spp.cfg',weights='weights/yolov3-spp-ultralytics.pt'):
    img_size = 512
    device = torch_utils.select_device(device='cpu')
    model = Darknet(cfg, img_size)
    model.load_state_dict(torch.load(weights, map_location=device)['model'])
    model.to(device).eval()
    return model

def letterbox(img, new_shape=(416, 416), color=(114, 114, 114),
              auto=True, scaleFill=False, scaleup=True, interp=cv2.INTER_AREA):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = max(new_shape) / max(shape)
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = new_shape
        ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=interp)  # INTER_AREA is better, INTER_LINEAR is faster
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def detect(image_path, model, conf_thres = 0.3):
    img0 = cv2.imread(image_path)
    img = letterbox(img0, new_shape=512)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
   
    img = torch.from_numpy(img).to(torch_utils.select_device(device='cpu'))
    img = img.float()
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    
    # Inference
    t1 = torch_utils.time_synchronized()
    pred = model(img)[0]
    t2 = torch_utils.time_synchronized()

    # Apply NMS
    pred = non_max_suppression(pred, 0.3, 0.6, multi_label=False)
    detections = []
    det = pred[0]
    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
    for *xyxy, conf, cls in det:
        detection = {
            'x1': int(xyxy[0]),
            'y1': int(xyxy[1]),
            'x2': int(xyxy[2]),
            'y2': int(xyxy[3]),
            'class' : int(cls)
        }
        if conf > conf_thres:
            detections.append(detection)
    return detections, img0
        
    


def draw_rectangle(event, x, y, flags, param):
    global x1, y1, x2, y2, drawing, display, running_state
    if not running_state:
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
        running_state = True
        break

cv2.waitKey(int(1000/60))


model = initialize_model()

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
    detections, display = detect(image_path, model)

    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), thickness=3)

    for detection in detections:
        if detection['class'] == 0:
            x_1, y_1, x_2, y_2 = detection['x1'], detection['y1'] , detection['x2'],detection['y2']
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
    base = os.path.basename(image_path)
    cv2.imwrite('./result/' + base,display)
    cv2.waitKey(int(1000/60))

    # break

cv2.destroyAllWindows()
