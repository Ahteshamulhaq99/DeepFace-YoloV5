from math import atan2, degrees, radians

def get_angle(point_1, point_2): #These can also be four parameters instead of two arrays
    angle = atan2(point_1[1] - point_2[1], point_1[0] - point_2[0])
    
    #Optional
    angle = degrees(angle)
    
  
    
    return angle


import cv2
from PIL import Image
import numpy as np
import torch
import detect_face
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weightPath = "./yolov5s-face.pt"

model = detect_face.load_model(weightPath, device)
img = cv2.imread("e.jpg")
w,h,_=img.shape
r=w/h
img=cv2.resize(img,(500,int(500*r)))
boxes,landmarks, confs = detect_face.detect_one(model, img, device)
# print("box>>",(boxes),"landmark>>>>",landmarks,"conf>>>", confs)
# box=[]
# landmark=[]
for i in range(len(boxes)):
# print(landmarks)img[int(y1):int(y2),int(x1):int(x2)]
    x1,y1,x2,y2=boxes[i]
    q1,r1,q2,r2=landmarks[i][:4]
    # print(x1,y1,x2,y2)
    angle=get_angle((q2,r2),(q1,r1))
    print(angle)
    # for i in range(5):
    #     point_x = int(landmarks[2 * i])
    #     point_y = int(landmarks[2 * i + 1])
    #     cv2.circle(img, (point_x, point_y), i, (255,39,30), -1)
    img2 = Image.fromarray(img[int(y1):int(y2),int(x1):int(x2)])
    img2=np.array(img2.rotate(angle))
    cv2.imshow("image", img2)


    cv2.waitKey(0)

cv2.destroyAllWindows()
