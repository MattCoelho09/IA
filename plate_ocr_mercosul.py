from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] 
from utils.plots import Annotator, colors
from utils.general import (LOGGER, check_img_size,
                           increment_path, non_max_suppression,  scale_coords)
from utils.torch_utils import select_device, time_sync
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams ,LoadImagesOcr
from models.common import DetectMultiBackend

import torch
import cv2
import numpy as np
# import uuid
class PlateOcrMercosul:
    def __init__(self):
        self.view_img = ''
        self.img_size_param = 640
        self.conf_thres = 0.55
        self.iou_thres = 0.45
        self.opt_classes = None
        self.opt_agnostic_nms = False
        self.line_thickness=3,
        self.augment=False,  # augmented inference
        self.classes=None, 
        self.agnostic_nms=False
        self.max_det=1000
        self.weights = 'weights/exp11/best.pt'
        self.dnn=False,
        self.data= 'data/ocr_mercosul.yaml',
        self.imgsz=640, 
        self.half=False,
        self.model = DetectMultiBackend('weigth_types/exp67/weights/best.pt', device=select_device('cpu'), dnn=self.dnn, data=self.data[0])
    
    def Tratar(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (640, 640), interpolation=4)
        img = cv2.bitwise_not(img)
        img = cv2.GaussianBlur(img, (7, 7), 0)
        kernel = np.ones((5, 5), np.uint8)
        # img = cv2.dilate(img, kernel, iterations=6)
        img = cv2.erode(img, kernel, iterations=1)  # best 7
        img = cv2.threshold(img, 1, 244, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        cv2.adaptiveThreshold(cv2.GaussianBlur(img, (5, 5), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31,2)
        cv2.adaptiveThreshold(cv2.bilateralFilter(img, 9, 75, 75), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,31, 2)
        cv2.adaptiveThreshold(cv2.medianBlur(img, 1), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img

    def ReadPlate(self, img):
        img = self.Tratar(img)
        array_teste = []
        plate_text=''
        # # cv2.imshow('DENTRO', img)
        # print("Entrei")
        # # cv2.waitKey(100)
        device = select_device('cpu')
        stride, names, pt, jit, _, _ = self.model.stride, self.model.names, self.model.pt, self.model.jit, self.model.onnx, self.model.engine
        imgsz = check_img_size(self.imgsz, s=stride)  # check image size

        # Half
        self.half =False
        if pt or jit:
            self.model.model.half() if self.half else self.model.model.float()
        dataset = LoadImagesOcr(img=img)
        
        bs = 1  # batch_size
        # vid_path, vid_writer = [None] * bs, [None] * bs

        # Run inference
        self.model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))
        dt, seen = [0.0, 0.0, 0.0], 0
        for path, im, im0s, vid_cap, s in dataset:
            t1 = time_sync()
            im = torch.from_numpy(im).to(device)
            im = im.half() if self.half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1
            
            # Inference
            visualize = False
            # detecta aqui plate_detect
            pred = self.model(im, augment=self.augment, visualize=visualize)
            t3 = time_sync()
            dt[1] += t3 - t2

            # NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes[0], self.agnostic_nms, max_det=self.max_det)
            dt[2] += time_sync() - t3

            for i, det in enumerate(pred):  # per image
                seen += 1
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                s += '%gx%g ' % im.shape[2:]  # print string
                annotator = Annotator(im0, line_width=self.line_thickness, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        # print(conf) #confidence
                        c = int(cls)  # integer class
                        print("word_detect",int(cls))
                        print(conf)  # confidence
                        label = names[c] 
                        c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                        array_teste.append((label, c1[0]))
                        # crop_img = im0[int(c1[1]):int(c2[1]), int(c1[0]):int(c2[0])]
                        # # cv2.imshow("crop", crop_img)
                        # # cv2.waitKey(100)
                        annotator.box_label(xyxy, label, color=colors(c, True))
                print(len(det))
                # cv2.imshow("vai", im0)
                # cv2.waitKey(0)

                #SALVA IMAGEM PARA RETREINO
                # if(plate_text!='PHX0F24'):plate_detect
                #     name = uuid.uuid4()
                #     cv2.imwrite("./save_to_train/{}.jpg".format(name), img)
            

                array_teste.sort(key=lambda tup:tup[1]) 
                for letra in array_teste:
                        plate_text = ''.join([plate_text, letra[0]])
            
                # if(plate_text!='PHX0F24'):
                #     name = uuid.uuid4()
                #     cv2.imwrite("../train_plates/{}.jpg".format(name), img)

            print(plate_text)
            return im0, plate_text