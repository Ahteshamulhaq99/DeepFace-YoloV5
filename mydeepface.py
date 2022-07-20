from deepface import DeepFace
import cv2
import numpy as np
import os
import torch
import detect_face
from PIL import Image
from tensorflow.keras.preprocessing import image
# import glob
from math import atan2, degrees, radians
models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib","SFace"]

model = DeepFace.build_model("Facenet512")

def detect_faceDefine(img, detector_backend = 'opencv', grayscale = False, enforce_detection = True, align = True):

	img_region = [0, 0, img.shape[0], img.shape[1]]

	#----------------------------------------------
	#people would like to skip detection and alignment if they already have pre-processed images
	if detector_backend == 'skip':
		return img, img_region

	#----------------------------------------------

	#detector stored in a global variable in FaceDetector object.
	#this call should be completed very fast because it will return found in memory
	#it will not build face detector model in each call (consider for loops)
	# face_detector = FaceDetector.build_model(detector_backend)

	try:
		detected_face, img_region = detect_faceDefine(face_detector, detector_backend, img, align)
	except: #if detected face shape is (0, 0) and alignment cannot be performed, this block will be run
		detected_face = None

	if (isinstance(detected_face, np.ndarray)):
		return detected_face, img_region
	else:
		if detected_face == None:
			if enforce_detection != True:
				return img, img_region
			else:
				raise ValueError("Face could not be detected. Please confirm that the picture is a face photo or consider to set enforce_detection param to False.")



def preprocess_face(img, target_size=(224, 224), grayscale = False, enforce_detection = True, detector_backend = 'opencv', return_region = False, align = False):

	#img might be path, base64 or numpy array. Convert it to numpy whatever it is.
	# img = load_image(img)
	base_img = img.copy()

	img,region= detect_faceDefine(img = img, detector_backend = "skip", grayscale = grayscale, enforce_detection = enforce_detection, align = align)

	#--------------------------

	if img.shape[0] == 0 or img.shape[1] == 0:
		if enforce_detection == True:
			raise ValueError("Detected face shape is ", img.shape,". Consider to set enforce_detection argument to False.")
		else: #restore base image
			img = base_img.copy()

	#--------------------------

	#post-processing
	if grayscale == True:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	#---------------------------------------------------
	#resize image to expected shape

	# img = cv2.resize(img, target_size) #resize causes transformation on base image, adding black pixels to resize will not deform the base image

	if img.shape[0] > 0 and img.shape[1] > 0:
		factor_0 = target_size[0] / img.shape[0]
		factor_1 = target_size[1] / img.shape[1]
		factor = min(factor_0, factor_1)

		dsize = (int(img.shape[1] * factor), int(img.shape[0] * factor))
        # print(img)
		img = cv2.resize(img, dsize)
        
		# Then pad the other side to the target size by adding black pixels
		diff_0 = target_size[0] - img.shape[0]
		diff_1 = target_size[1] - img.shape[1]
		if grayscale == False:
			# Put the base image in the middle of the padded image
			img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)), 'constant')
		else:
			img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2)), 'constant')

	#------------------------------------------
	try:
	#double check: if target image is not still the same size with target.
		if img.shape[0:2] != target_size:
			img = cv2.resize(img, target_size)
	except:
		pass
	#---------------------------------------------------

	#normalizing the image pixels

	img_pixels = image.img_to_array(img) #what this line doing? must?
	# print("1",img_pixels.shape)
	img_pixels = np.expand_dims(img_pixels, axis = 0)
	# print("2",img_pixels.shape)
	img_pixels /= 255 #normalize input in [0, 1]

	#---------------------------------------------------

	if return_region == True:
		return img_pixels
	else:
		return img_pixels





def get_angle(point_1, point_2): #These can also be four parameters instead of two arrays
    angle = atan2(point_1[1] - point_2[1], point_1[0] - point_2[0])

    angle = degrees(angle)
    return angle    

def findCosineDistance(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))


def process(img):
    width,height=160,160
    # img=cv2.resize(img,(width,height))
    # img=img.reshape(1,width,height,3)
    img=preprocess_face(img, target_size=(width, height), grayscale = False, enforce_detection = True, detector_backend = 'opencv', return_region = False, align = True)

    img_representation = model.predict(img)[0,:]
    return(img_representation)


def processDataset(dataset):
    known={}
    dir_list = os.listdir(dataset)
    
    for image in dir_list:
        # print("imgae",image)
        img=cv2.imread(dataset+image)
        print(img.shape)
        boxes, landmarks,confs = detect_face.detect_one(model1, img, device)
        print(boxes)
        if len(boxes)!=0:
            x1,y1,x2,y2=boxes[0]
            img=img[int(y1):int(y2),int(x1):int(x2)]
        # w,h,_=img.shape
        # r=h/w        
        # img=cv2.resize(img,(500,int(500*r)))        
        encoding=process(img)
        name=image.split(".")
        # print(name)
        known[name[0]] = encoding
    return known

def findDistances(img):

    distances={}
    # w,h,_=img.shape
    # r=h/w
    # img=cv2.resize(img,(500,int(500*r)))
    unknown_encoding=process(img)
    for known_encoding in known_encodings:
        # print("koooo",known_encoding)
        distance=findCosineDistance(unknown_encoding,known_encodings[known_encoding])
        # distances.append(distance)
        distances[known_encoding] = distance
    # print(distances)
    return distances

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weightPath = "./yolov5s-face.pt"

model1 = detect_face.load_model(weightPath, device)

known_encodings=processDataset(dataset="img/")
cap=cv2.VideoCapture(0)
while True:
    # img=cv2.imread("2.jpg")
    ret,img=cap.read()
    # w,h,_=img.shape
    # r=w/h
    # img=cv2.resize(img,(1000,int(1000*r)))
    # img = cv2.flip(img, 1)
    boxes,landmarks, confs = detect_face.detect_one(model1, img, device)
    for i in range(len(boxes)):        
        
        x1,y1,x2,y2=boxes[i]
        q1,r1,q2,r2=landmarks[i][:4]
        x1=int(x1)
        y1=int(y1)
        x2=int(x2)
        y2=int(y2)
        # print(x1,y1,x2,y2)
        angle=get_angle((q2,r2),(q1,r1))
        # print(angle)
        img2 = Image.fromarray(img[int(y1):int(y2),int(x1):int(x2)])
        img2=np.array(img2.rotate(angle))     

        result=findDistances(img2)
       
        person="unknown"
        person=min(result, key=result.get)
        # print(min(result, key=result.get))
        
        if result[person]<=0.45:
            print(person,">>>",result)
            cv2.putText(img, person, (x1,y1), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0), 2, cv2.LINE_AA)
        else:
            print(person," >>> ",result)
            person="unknown"
            cv2.putText(img, person, (x1,y1), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.rectangle(img, (x1,y1), (x2,y2), (0,200,29), 3)
    cv2.imshow("img",img)
    k=cv2.waitKey(1)
    if k==27:
        break
cv2.destroyAllWindows()
       


