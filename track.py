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
from centroid_direction import CentroidTracker
from centroid_direction import TrackableObject
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet

from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import LoadImages, LoadStreams, VID_FORMATS
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors, save_one_box
import numpy as np
from collections import Counter
def box_area(arr):
    # arr: np.array([[x1, y1, x2, y2]])
    width = arr[:, 2] - arr[:, 0]
    height = arr[:, 3] - arr[:, 1]
    return width * height

def _box_inter_union(arr1, arr2):
    # arr1 of [N, 4]
    # arr2 of [N, 4]
    area1 = box_area(arr1)
    area2 = box_area(arr2)

    # Intersection
    top_left = np.maximum(arr1[:, :2], arr2[:, :2]) # [[x, y]]
    bottom_right = np.minimum(arr1[:, 2:], arr2[:, 2:]) # [[x, y]]
    wh = bottom_right - top_left
    # clip: if boxes not overlap then make it zero
    intersection = wh[:, 0].clip(0) * wh[:, 1].clip(0)

    #union 
    union = area1 + area2 - intersection
    return intersection, union

def box_iou(arr1, arr2):
    # arr1[N, 4]
    # arr2[N, 4]
    # N = number of bounding boxes

    inter, union = _box_inter_union(arr1, arr2)
    iou = inter / union
    return iou

def buildpt(Point_top,point_bot):
    Vector_x=(-(point_bot[0]-Point_top[0])/(point_bot[1]-Point_top[1]),1)
    return Vector_x[1], Vector_x[0],-Vector_x[1]*Point_top[0]-Vector_x[0]*Point_top[1]


trackableObjects = {}
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
# DeepSort Parameters Definition
max_cosine_distance = 0.3
nn_budget = None
nms_max_overlap = 1.0

peopleOut = 0
peopleIn = 0

peopleOuttop = 0
peopleIntop = 0
peopleOutbot = 0
peopleIntbot = 0
# DeepSort Loading
model_filename = 'model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename,batch_size=1)

metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)


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

source='/home/quang/Documents/CountPeople/People-Detection-and-Counting-In-Out-Line/01.mp4'
source="/home/quang/Documents/CountPeople/output1.mp4"
yolo_model='/home/quang/Documents/CountPeople/Yolov5_DeepSort_Pytorch/yolov5x.pt'
# yolo_model='/home/quang/Documents/CountPeople/Yolov5_DeepSort_Pytorch/crowdhuman_yolov5m.pt'

model = DetectMultiBackend(yolo_model, device=device, dnn=opt.dnn)
stride, names, pt = model.stride, model.names, model.pt
imgsz=400
imgsz = check_img_size(imgsz, s=stride)  # check image size
# print("imgsz",imgsz)
dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
nr_sources = 1
dt, seen = [0.0, 0.0, 0.0, 0.0], 0
H,W=None, None
color = (255, 0, 0)
                # Line thickness of 2 px
thickness = 2
vehice_in={}
vehice_out={}
info=[]
dict_vehice={}
out=None
line1_x=(499,86)
line1_y=(766,326)
line2_x=(122,67)
line2_y=(398,467)

for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
    frame=im0s.copy()
    t1 = time_sync()
    im = torch.from_numpy(im).to(device)

    if W is None or H is None:
	    (H, W) = im0s.shape[:2]

    if out is None:
        out = cv2.VideoWriter('outpy2.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 25, (300,300))

    im =im/ 255.0 
    # im = im.half() if half else im.float()  # uint8 to fp16/32
     # 0 - 255 to 0.0 - 1.0
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
            xywhs=xywhs.cpu()
            confs = det[:, 4]
            clss = det[:, 5]
            boxs=[]
            boxs_ss=[]
            for i_box in xywhs:
                x_c,y_c,w,h=i_box
                x1=x_c-w/2
                y1=y_c-h/2
                boxs.append([x1,y1,w,h])
            for i_box in xywhs:
                x_c,y_c,w,h=i_box
                x1=x_c-w/2
                y1=y_c-h/2
                boxs_ss.append([x1,y1,x1+w,y1+h])

            # for box in boxs:
            #     c2 = (int(box[1]) + int(box[3]))/2
            #     color = (255, 0, 0)
            #     # Line thickness of 2 px
            #     thickness = 2
            #     # start_point = (int(box[0] ), int(box[1]))
            #     # Ending coordinate, here (220, 220)
            #     # represents the bottom right corner of rectangle
            #     # end_point = (int(box[2]),int(box[3]))
            #     start_point = (int(box[0]-int(box[2])/2 ), int(box[1]-box[3]/2))
            #     # Ending coordinate, here (220, 220)
            #     # represents the bottom right corner of rectangle
            #     end_point = (int(box[0])+int(box[2]/2),int(box[1])+int(box[3]/2))
            #     # Blue color in BGR

            #     im0 = cv2.rectangle(im0, start_point, end_point, color, thickness)
            # cv2.imshow('1', im0)
            # cv2.waitKey(1) 
        
            features = encoder(im0,xywhs)

            detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
            
            
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
       
            detections = [detections[i] for i in indices]
          
            tracker.predict()
            tracker.update(detections, H)
            

            image = cv2.line(frame, line1_x, line1_y, color, thickness)
            image = cv2.line(frame, line2_x, line2_y, color, thickness)

            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue 
                bbox = track.to_tlbr()
                # print(bbox)
                c1 = (int(bbox[0]) + int(bbox[2]))/2
                c2 = (int(bbox[1]) + int(bbox[3]))/2

                centerPoint = (int(c1), int(c2))
                centerPoint = (int(bbox[0]), int(c2))
                # start_point = (int(box[0] ), int(box[1]))
                # Ending coordinate, here (220, 220)
                # represents the bottom right corner of rectangle
                # end_point = (int(box[2]),int(box[3]))
                start_point = (int(bbox[0]), int(bbox[1]))
                # Ending coordinate, here (220, 220)
                # represents the bottom right corner of rectangle
                end_point = (int(bbox[2]),int(bbox[3]))
                # Blue color in BGR

                frame = cv2.rectangle(frame, start_point, end_point, color, thickness)
                max_iou=0
                max_id=0
                
                for i in range(len(boxs_ss)):
                    box1=np.array([[int(i.item()) for i in boxs_ss[i]]])
                    box2= np.array([[int (i) for i in bbox]])
                    if box_iou(box1,box2) > max_iou:
                        max_iou=box_iou(box1,box2)
                        max_id=i
                c = int(clss[max_id])  # integer class
                label = f'{names[c]} '
                try:
                    dict_vehice[track.track_id].append(c)
                except:
                    dict_vehice[track.track_id]=[]
                    dict_vehice[track.track_id].append(c)

                abc= dict(Counter(dict_vehice[track.track_id]))
                
                max_id= list(abc.keys())[0]
                # c = int(clss[max_id])  # integer class
                label = f'{names[max_id]} '
                cv2.putText(frame, str(track.track_id),centerPoint,0, 5e-3 * 200, (0,0,255),2)
                cv2.putText(frame, str(label),centerPoint,0, 5e-3 * 200, (0,0,255),2)
                cv2.circle(frame, centerPoint, 4, (0, 0, 255), -1)
                startY, endY = int(bbox[1]), int(bbox[3])
                startX, endX = int(bbox[0]), int(bbox[2])
                XMidT1 = int(endX)
                yMidT1 = int((endY + startY)/2)

                XMidT = int(startX)
                yMidT = int((startY+endY)/2)

                a,b,c=buildpt(line1_x,line1_y)
                

                # if track.stateOutMetro['top'] == 1 and (a*XMidT+b*yMidT+c < 0) and track.noConsider == False:
                #     peopleOuttop += 1 
                #     track.stateOutMetro['top'] = 0
                #     track.noConsider = True
                #     cv2.line(frame, (0, H // 2 +50), (W, H // 2 +50), (0, 0, 0), 2)


                # c = int(clss)  # integer class
                # label = f'{id} {names[c]} '

                if track.stateOutMetro['top'] == 0 and (a*XMidT1+b*yMidT1+c >= 0) and track.noConsider == False:
                    
                    try:
                        vehice_in[label] += 1 
                    except:
                        vehice_in[label]=1
                    track.stateOutMetro['top'] = 1

                    track.stateOutMetro['bot'] = 1
                    track.noConsider = True


                    # cv2.line(frame, (0, H // 2 +50), (W, H // 2 +50), (0, 0, 0), 2)
                cv2.circle(frame, (XMidT1,yMidT1), 10, (0, 0, 255), -1)
                cv2.circle(frame, (XMidT,yMidT), 10, (0, 255, 255), -1)
                a,b,c=buildpt(line2_x,line2_y)
                if track.stateOutMetro['bot'] == 1 and (a*XMidT+b*yMidT+c < 0) and track.noConsider == False:
                    try:
                        vehice_out[label] += 1 
                    except:
                        vehice_out[label]=1
                    # peopleOutbot[label] += 1 
                    track.stateOutMetro['top'] = 0

                    track.stateOutMetro['bot'] = 0
                    
                    track.noConsider = True


                    # cv2.line(frame, (0, H // 2 +50), (W, H // 2 +50), (0, 0, 0), 2)
               

                # if track.stateOutMetro['bot'] == 0 and (a*XMidT+b*yMidT+c >= 0) and track.noConsider == False :
                #     peopleIntbot += 1
                #     track.stateOutMetro['bot'] = 1
                #     track.noConsider = True
                #     cv2.line(frame, (0, H // 2 +50), (W, H // 2 +50), (0, 0, 0), 2
                #     )


            # cv2.line(frame, (0, H // 2 +50), (W, H // 2 + 50), (0, 0, 255), 2)
    print(vehice_in)
    info_in=[]
    info_out=[]
    for i in vehice_in:
        info1 = ( str(i),vehice_in[i])
        info_in.append(info1)

    for i in vehice_out:
        info1 = ( str(i),vehice_out[i])
        info_out.append(info1)

    frame = cv2.line(frame, line1_x, line1_y, color, thickness)
    frame = cv2.line(frame, line2_x, line2_y, color, thickness)

    for (i, (k, v)) in enumerate(info_in):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (W-300, 30+ ((i * 20) + 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2)
        # break

    for (i, (k, v)) in enumerate(info_out):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2)
            # break
    
    # cv2.SetMouseCallback('real image', on_mouse, 0)   
    cv2.imshow('', frame)
    # out.write(frame)
    frame1=cv2.resize(frame,(300,300))
    out.write(frame1)
    cv2.waitKey(1) 
        


            # for j, (output,conf,cls )in enumerate(zip(xywhs, confs,clss)):
            #     bboxes = output[0:4]
            #     id = conf
            #     cls = cls
            #     c = int(cls)  # integer class
            #     label = f'{id} {names[c]} {conf:.2f}'
            #     # start_point = (5, 5)
            #     # # Ending coordinate, here (220, 220)
            #     # # represents the bottom right corner of rectangle
            #     # end_point = (220, 220)
                # start_point = (int(bboxes[0]-int(bboxes[2])/2 ), int(bboxes[1]-bboxes[3]/2))
                # # Ending coordinate, here (220, 220)
                # # represents the bottom right corner of rectangle
                # end_point = (int(bboxes[0])+int(bboxes[2]/2),int(bboxes[1])+int(bboxes[3]/2))
                # # Blue color in BGR
            #     color = (255, 0, 0)
            #     # Line thickness of 2 px
            #     thickness = 2
            #     # Using cv2.rectangle() method
            #     # Draw a rectangle with blue line borders of thickness of 2 px
            #     im0 = cv2.rectangle(im0, start_point, end_point, color, thickness)

                # print(bboxes)
                # annotator.box_label(bboxes, label, color=colors(c, True))
    LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s)')
    # if True:
    #     cv2.imshow(str(p), im0)
    #     cv2.waitKey(1)  # 1 millisecond