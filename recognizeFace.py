# import libraries
import detect_face
import numpy
import cv2
import torch
from queue import Queue
import glob
import threading
import time
# import api
import io
import numpy as np
import os
from deepface import DeepFace
# from deep_sort.utils.parser import get_config
# from deep_sort.deep_sort import DeepSort

class recognizer():
    def __init__(self, model, model_rec, detector='DeepFace', dataset=''):
        self.detector = detector
        self.dataset = dataset
        self.model_rec = model_rec
        self.model = model
        self.known_encodings = None

    def findCosineDistance(self,  source_representation, test_representation):
        a = np.matmul(np.transpose(source_representation), test_representation)
        b = np.sum(np.multiply(source_representation, source_representation))
        c = np.sum(np.multiply(test_representation, test_representation))
        return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

    def process(self, img):
        width, height, _ = img.shape
        # img=cv2.resize(img,(width,height))
        img = img.reshape(1, width, height, 3)
        img_representation = self.model_rec.predict(img)[0,:]
        return(img_representation)

    def processDataset(self):
        known={}
        dir_list = os.listdir(self.dataset)
        
        for image in dir_list:
            # print("imgae",image)
            img=cv2.imread(os.path.join(self.dataset, image))
            # boxes, confs = detect_face.detect_one(self.model, img, device)
            
            # x1,y1,x2,y2=boxes[0]
            
            # cv2.imwrite(os.path.join(self.dataset, "face_"+image), img[int(y1):int(y2),int(x1):int(x2)])
            encoding = self.process(img)
            name=image.split(".")
            # print(name)
            known[name[0]] = encoding

        
        self.known_encodings = known
        return known

    def findDistances(self, img):
        distances={}
        
        unknown_encoding = self.process(img)
        if  self.known_encodings is None:
            self.known_encodings = self.processDataset()

        for known_encoding in self.known_encodings:
            # print("koooo",known_encoding)
            distance=self.findCosineDistance(unknown_encoding, self.known_encodings[known_encoding])
            # distances.append(distance)
            distances[known_encoding] = distance
        # print(distances)
        
        person=min(distances, key=distances.get)
        print(person)
        return distances

def trackFaces(deepsort, confs, xywhs, clss, img):
    # pass detections to deepsort
    outputs = deepsort.update(xywhs, confs, clss, img)
    deepsort.increment_ages()
    print(outputs)
    return outputs


def detect_faces(model):
    cap=cv2.VideoCapture(0)

    # cfg = get_config()
    # cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
    # deepsort = DeepSort('osnet_x0_25', max_dist=cfg.DEEPSORT.MAX_DIST,max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
    #                     max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET, use_cuda=True)

    while True:
        ret,img = cap.read()
        # img=cv2.flip(img,-1)
        # img=img[:800,100:1600]
        # img=cv2.resize(img,(200,200))
        boxes, confs = detect_face.detect_one(model, img, device)
        
        # tracked = trackFaces(deepsort, confs, np.array(boxes), [0], img)
        # print(tracked)

        if len(boxes)>0:
            qfaces.put((boxes, img))
        window_name = "frame"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow('frame', img)
        if cv2.waitKey(1) & 0xFF == 27:
            cv2.destroyAllWindows()
            break


def recognise_faces(faceRecognizer):
    # (face_ids, names) = api.get_approved_faces()
    
    
    while True:
        print(qfaces.qsize())
        boxes, img = qfaces.get()
        for x1,y1,x2,y2 in boxes: 
            x1=int(x1)
            y1=int(y1)
            x2=int(x2)
            y2=int(y2)

            result = faceRecognizer.findDistances(img[int(y1):int(y2),int(x1):int(x2)])
            # print(result)


        # ret, buf = cv2.imencode('.jpeg', img)
        # stream = io.BytesIO(buf)
        # print("API CALLED")
        # api.compare(face_ids, names, stream,)



if __name__ == "__main__":
    qfaces = Queue(maxsize=10)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ",device)
    weightPath = "./yolov5s-face.pt"

    model = detect_face.load_model(weightPath, device)
    model_rec = DeepFace.build_model("SFace")
    faceRecognizer = recognizer(model, model_rec, detector='DeepFace', dataset="img/")
    faceRecognizer.processDataset()
    
    t1 = threading.Thread(target=detect_faces, args=(model,))
    t2 = threading.Thread(target=recognise_faces,  args=(faceRecognizer,))

    t1.start()
    t2.start()
