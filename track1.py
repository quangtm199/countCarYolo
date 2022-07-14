import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import LoadImages, LoadStreams, VID_FORMATS
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors, save_one_box
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
device = select_device('cpu')
parser = argparse.ArgumentParser()
# parser.add_argument('--augment', action='store_true', help='augmented inference')
# parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
parser.add_argument('--augment', action='store_true', help='augmented inference')
parser.add_argument('--update', action='store_true', help='update all models')
parser.add_argument('--evaluate', action='store_true', help='augmented inference')
parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
parser.add_argument('--visualize', action='store_true', help='visualize features')
parser.add_argument('--max-det', type=int, default=1000, help='maximum detection per image')
parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
parser.add_argument('--name', default='exp', help='save results to project/name')
parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
  
opt = parser.parse_args()

source='/home/quang/Documents/CountPeople/01.mp4'
yolo_model='/home/quang/Documents/CountPeople/Yolov5_DeepSort_Pytorch/yolov5x.pt'
model = DetectMultiBackend(yolo_model, device=device, dnn=opt.dnn)
stride, names, pt = model.stride, model.names, model.pt
imgsz=640*2
imgsz = check_img_size(imgsz, s=stride)  # check image size
dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
nr_sources = 1
dt, seen = [0.0, 0.0, 0.0, 0.0], 0
for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
    t1 = time_sync()

    im = torch.from_numpy(im).to(device)
    # im = im.half() if half else im.float()  # uint8 to fp16/32
    im =im/ 255.0  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    t2 = time_sync()
    dt[0] += t2 - t1
    # visualize = increment_path("" / Path(path[0]).stem, mkdir=True) if opt.visualize else False
    pred = model(im, augment=opt.augment, visualize="")
    t3 = time_sync()
    dt[1] += t3 - t2
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms, max_det=opt.max_det)

    dt[2] += time_sync() - t3
    for i, det in enumerate(pred):
        p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
        annotator = Annotator(im0, line_width=2, pil=not ascii)
        if det is not None and len(det):
           
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
            xywhs = xyxy2xywh(det[:, 0:4])
            confs = det[:, 4]
            clss = det[:, 5]
            
            for j, (output,conf,cls )in enumerate(zip(xywhs, confs,clss)):
                bboxes = output[0:4]
                id = conf
                cls = cls
                c = int(cls)  # integer class
                label = f'{id} {names[c]} {conf:.2f}'
                # start_point = (5, 5)
                # # Ending coordinate, here (220, 220)
                # # represents the bottom right corner of rectangle
                # end_point = (220, 220)
                start_point = (int(bboxes[0]-int(bboxes[2])/2 ), int(bboxes[1]-bboxes[3]/2))
                # Ending coordinate, here (220, 220)
                # represents the bottom right corner of rectangle
                end_point = (int(bboxes[0])+int(bboxes[2]/2),int(bboxes[1])+int(bboxes[3]/2))
                # Blue color in BGR
                color = (255, 0, 0)
                # Line thickness of 2 px
                thickness = 2
                # Using cv2.rectangle() method
                # Draw a rectangle with blue line borders of thickness of 2 px
                im0 = cv2.rectangle(im0, start_point, end_point, color, thickness)

                # print(bboxes)
                # annotator.box_label(bboxes, label, color=colors(c, True))
    LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s)')
    if True:
        cv2.imshow(str(p), im0)
        cv2.waitKey(1)  # 1 millisecond