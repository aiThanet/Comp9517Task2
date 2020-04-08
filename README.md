
# COMP9517 20T1 Group Project : Part2
user can draw a rectangular bounding box within the whole window. The system will report
- pedestrians who enter the bounding box
- pedestrians who move out of the bounding box

## How to run:
1. run script `./weights/download_yolov3_weights.sh` to download pre-trained weights (around 200MB)
1. Place all images in following path -> Group_Component/seqeunce/*.jpg
1. Run `python Task2.py`
1. check `python Task2.py -h` for custom argument

## After run:
1. program will ask you to draw the bounding box
1. After drawing the bounding box, press Enter


## Reference
This project use YOLOv3-SPP model implemented by ultraltyics team.
![Github](https://github.com/ultralytics/yolov3)

