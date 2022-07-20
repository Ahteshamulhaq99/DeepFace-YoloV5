# from matplotlib import image
# import api
import detect_face
import cv2
import time
from threading import Thread
import threading
import torch
import os 
from queue import Queue


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")
print("device: ",device)
weightPath = "./yolov5s-face.pt"

model = detect_face.load_model(weightPath, device)




name = ""
prev_name = ""
name1 = ""
prev_name1 = ""
start_time = time.time()
end_time = time.time()

# cap =cv2.VideoCapture('rtsp://12345najam:12345najam@192.168.0.100:554/stream1')


start =time.time()

import mysql.connector
from datetime import datetime
mydb = mysql.connector.connect( host="localhost",
user="root",
password="",
database="attendencedb"
)

def myDatabase(name):

	now = datetime.now()
	dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
	mycursor = mydb.cursor(buffered=True)
	# print("dblabel++++++++++++++++++====",label)

	# if mycursor.execute("SELECT * FROM enterence WHERE (name) =( %s)",(label)):
	# 	mycursor.execute('insert into enterence(name,time) values(%s, %s)', (label,dt_string))
	
	# else:
	# 	pass

	query = "SELECT * FROM `enterence` WHERE name = '{}';".format(name)
	mycursor.execute(query)
	if len([i for i in mycursor])==0:
	# mycursor.execute(f'INSERT INTO PERSON (name,time) VALUES ("ali",{d})')
		mycursor.execute('insert into enterence(name,time) values(%s, %s)', (name,dt_string))
	# all=mycursor.execute("select * from enterence")
	
	
	# print("Num of Rows: ",len([i for i in mycursor]))
	# for name in mycursor:
	# 	print("NAME ", name[1])

	# print("alllllll",all)
	
	mydb.commit()
	mycursor.close()


    ###################################################################


def myDatabase1(name1):

	now = datetime.now()
	dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
	mycursor = mydb.cursor(buffered=True)
	# print("dblabel++++++++++++++++++====",label)

	# if mycursor.execute("SELECT * FROM enterence WHERE (name) =( %s)",(label)):
	# 	mycursor.execute('insert into enterence(name,time) values(%s, %s)', (label,dt_string))
	
	# else:
	# 	pass

	query = "SELECT * FROM `exit1` WHERE name = '{}';".format(name1)
	mycursor.execute(query)
	if len([i for i in mycursor])==0:
	# mycursor.execute(f'INSERT INTO PERSON (name,time) VALUES ("ali",{d})')
		mycursor.execute('insert into exit1(name,time) values(%s, %s)', (name1,dt_string))
	# all=mycursor.execute("select * from enterence")
	
	
	# print("Num of Rows: ",len([i for i in mycursor]))
	# for name in mycursor:
	# 	print("NAME ", name[1])

	# print("alllllll",all)
	
	mydb.commit()
	mycursor.close()


def enter_cam():
    camName = 'Entrance'
    (face_ids, names) = api.get_approved_faces()
    cap =cv2.VideoCapture(1)
    # cap =cv2.VideoCapture(0)


    while True: 
        ret,im  = cap.read() 
        im=cv2.resize(im,(400,200))
        if ret==False:
            continue 

        end_time=time.time() -start
        boxes, confs = detect_face.detect_one(model, im, device)
        
        if len(boxes)>0:
            qfaces.put((camName, boxes, confs, im))
    
        else:
            pass
        cv2.imshow('enterance',im)
        # return name

            
        if cv2.waitKey(1) & 0xFF == 27:
            break



def exit_cam(): 
    camName = "Exit"
    cap =cv2.VideoCapture(0)

    while True: 
        ret1,im1  = cap.read() 
        im1=cv2.resize(im1,(400,200))
        if ret1==False:
            continue 

        end_time=time.time() -start
        
        
        boxes, confs = detect_face.detect_one(model, im1, device)
        
        if len(boxes)>0:
            qfaces.put((camName, boxes, confs, im1))
            # if int(end_time) > 2:
            #     start_time = time.time()
            #     # try:
            #     cv2.imwrite("temp1.jpg",im1)

        cv2.imshow('exit',im1)
        # return name1

            
        if cv2.waitKey(1) & 0xFF == 27:
            break



        ################################

def recognizeFace():
    (face_ids, names) = api.get_approved_faces()
    while True:
        camName, boxes, confs, im1 = qfaces.get()
        
        name=api.compares(face_ids, names)
        
        print("name_found in camera "+camName, name)

        myDatabase1(name)


if __name__ == "__main__":
    # creating thread
    qfaces = Queue(maxsize=10)
    # qfaces1 = Queue(maxsize=10)


    t1 = threading.Thread(target=enter_cam )
    t2 = threading.Thread(target=exit_cam )
    t3 = threading.Thread(target=recognizeFace)
    # starting thread 1
    t1.start()
    # starting thread 2
    t2.start()
    t3.start()
  
  
    # both threads completely executed
    print("Done!")

