import cv2
import numpy as np
from imutils.video import VideoStream
from PIL import Image,ImageFont, ImageDraw
import time
import imutils

from detector.model_cpu.darknet_opencv import Darknet_OpenCV_YOLO
from detector.utils.draw_rect import Drawing_obj

#USING YOLO KERAS
yolo = Darknet_OpenCV_YOLO("detector")
#DRAWER
drawer = Drawing_obj("detector/data/config/coco.names",'detector/utils/font/FiraMono-Medium.otf')

class_list = ["person"]
source = 0


vs = VideoStream(source).start()
time.sleep(2.0)

while(True):
    image = vs.read()
    image = imutils.resize(image, 640,480)

    if image is not None:
        image, detect = yolo.detect(image, confidence_val=0.001, class_list = class_list, size_min = 0.01, size_max = 1)
    
        for obj in detect:
            cx = obj[0]
            cy = obj[1]

        image = Image.fromarray(np.uint8(image.copy()))
        for (cx,cy, w, h, ind_class, score) in detect:
            x1 = cx - w//2
            x2 = cx + w//2
            y1 = cy - h//2
            y2 = cy + h//2
            drawer.draw(image,x1,y1,x2,y2, ind_class, "jsd")

        result = np.array(image)

        cv2.imshow('Frame',result)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else: 
        break

vs.stop()
cv2.destroyAllWindows()