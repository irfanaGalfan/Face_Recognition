import cv2 #For Image processing 
import numpy as np #For converting Images to Numerical array 
import os#To handle directories 
from PIL import Image,ImageTk #Pillow lib for handling images
import tkinter as tk
import smtplib,ssl
import yagmail
import RPi.GPIO as GPIO
import mediapipe as mp
from time import sleep
from guizero import App, Text, PushButton
from gpiozero import Robot, LED
import pandas as pd
from datetime import datetime
from subprocess import call
from gpiozero import DistanceSensor
import face_recognition
Button = 23
Motor1A=4
Motor1B=5
#sensor = DistanceSensor(echo=17, trigger=14)
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(Motor1A,GPIO.OUT)
GPIO.setup(Motor1B,GPIO.OUT)
servo_pin = 13
GPIO.setup(servo_pin,GPIO.OUT)
pwm = GPIO.PWM(servo_pin,50)
GPIO.setup(Button,GPIO.IN,pull_up_down=GPIO.PUD_UP)
GPIO.output(Motor1A,GPIO.LOW)
GPIO.output(Motor1B,GPIO.LOW)
mp_drawing = mp.solutions.drawing_utils
mp_face = mp.solutions.face_detection.FaceDetection(model_selection=1,min_detection_confidence=0.5)

def button():
    button_state=GPIO.input(Button)
    print(button_state)
    if button_state==0:
            #GPIO.output(LED,GPIO.HIGH)
        print("button on")
        call(["espeak","HI I AM SPIRIT"+" SMART AND PRECIDE INTELLIGENT RECOGNITION TECHNOLOGY"])
                
    else:
        print("button off")
        call(["espeak","HI I AM SPIRIT"+" SMART AND PRECIDE INTELLIGENT RECOGNITION TECHNOLOGY"])
        sleep(1)
button()
width=680
height=480
top=tk.Tk()
top.title("WELCOME TO FACE RECOGNITION")
top.geometry('1024x768')
image1 = Image.open("/home/iisajman/Desktop/Face-Recognition/Untitled.jpg")
test = ImageTk.PhotoImage(image1)
label1 = tk.Label(image=test)
# label1.image = test
label1.pack(expand=True)# Position image
def face_recog1():
    now1= datetime.now() 
    # dd/mm/YY H:M:S
    dt_string1 = now1.strftime("%d-%m-%Y %H:%M:%S")
    dt_string2=str(dt_string1)
    
    video_capture = cv2.VideoCapture(0)
    d1="/home/iisajman/Desktop/Face-Recognition/Footages/"
    videopath=os.path.join(d1,dt_string2)
    os.mkdir(videopath)
    result = cv2.VideoWriter(os.path.join(videopath,'Output.avi'), cv2.VideoWriter_fourcc(*'MJPG'),10,(640, 480))

    #imghumaira = face_recognition.load_image_file('/home/iisajman/Desktop/Face-Recognition/Face_Images/humaira.jpeg')
    #imghumaira_encoding = face_recognition.face_encodings(imghumaira)[0]
    
    imgirfana = face_recognition.load_image_file('/home/iisajman/Desktop/Face-Recognition/Face_Images/irfana3.1.jpeg')
    imgirfana_encoding = face_recognition.face_encodings(imgirfana)[0]

    imgobama = face_recognition.load_image_file('/home/iisajman/Desktop/Face-Recognition/Face_Images/obama.jpg')
    imgobama_encoding = face_recognition.face_encodings(imgobama)[0]
    imgArshath = face_recognition.load_image_file('/home/iisajman/Desktop/Face-Recognition/Face_Images/Arshath4.2.jpeg')
    imgArshath_encoding = face_recognition.face_encodings(imgArshath)[0]
    
    imgVarun = face_recognition.load_image_file('/home/iisajman/Desktop/Face-Recognition/Face_Images/Varun5.1.jpeg')
    imgVarun_encoding = face_recognition.face_encodings(imgVarun)[0]
    
    imgMonish = face_recognition.load_image_file('/home/iisajman/Desktop/Face-Recognition/Face_Images/monish6.2.jpeg')
    imgMonish_encoding = face_recognition.face_encodings(imgMonish)[0]

    # Create arrays of known face encodings and their names
    known_face_encodings = [
#         imghumaira_encoding,
        imgobama_encoding,
        imgirfana_encoding,
        imgArshath_encoding,
        imgVarun_encoding,
        imgMonish_encoding,
        
    ]
    known_face_names = [
        #"Humaira",
        "Obama",
        "Irfana",
        "Arshath",
        "Varun",
        "Monish"
        
    ]

    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()
        result.write(frame)

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                # # If a match was found in known_face_encodings, just use the first one.
                # if True in matches:
                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)

        process_this_frame = not process_this_frame


        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            #call(["espeak","WELCOME"+name])
            if name=="Unknown":
#                 crop_face = frame[y:y+h, x:x+w]
                call(["espeak","SORRY BETTER LUCK NEXT TIME"])
                crop_face = frame[right:right+left, top:top+bottom]
                image = Image.fromarray(crop_face, "RGB")
                image.save('1.jpg')
                receiver = "irfanap@iisajman.org"  # receiver email address
                body = "Unauthorized Entry is Found!!!"  # email body
                filename = "/home/iisajman/Desktop/Face-Recognition/1.jpg" 
                yag = yagmail.SMTP("irfanazahir@gmail.com", "zbidzubjjvwnygvu")
                yag.send(
                    to=receiver,
                    subject="Unauthorized Entry", 
                    contents=body, 
                    attachments=filename,)
                now = datetime.now() 
                    # dd/mm/YY H:M:S
                dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                df=pd.read_csv('/home/iisajman/Desktop/Face-Recognition/log.csv')
                df1=pd.DataFrame([[dt_string,"unrecognized"]],columns=["TimeStamp","Name"])
                df2=df.append(df1)
                #print(df)
                #print(df1)
                df2.to_csv('/home/iisajman/Desktop/Face-Recognition/log1.csv',header=True,index=False)
            elif name!="UnKnown":
                now = datetime.now() 
                # dd/mm/YY H:M:S
                dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                df=pd.read_csv('/home/iisajman/Desktop/Face-Recognition/log.csv')
                df1=pd.DataFrame([[dt_string,name]],columns=["TimeStamp","Name"])
                df2=df.append(df1)
                #print(df)
                #print(df1)
                df2.to_csv('/home/iisajman/Desktop/Face-Recognition/log.csv',header=True,index=False)
                call(["espeak","hi"+name]) 
        # Display the resulting image
        #image=cv2.flip(frame,0)
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()
    
def facedetection():
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    #cap.set(3,640) # set Width
    #cap.set(4,480) # set Height
    while True:
        ret, img = cap.read()
        #img = cv2.flip(img, -1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=5,minSize=(20,20))
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
        cv2.imshow('video',img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
               break
    cap.release()
    cv2.destroyAllWindows()
    
def snapshot():
    def get_input():
        sid=my_text_box.get("1.0","end-1c")
        print(sid)
        directory=my_text_box1.get("1.0","end-1c")
        print(directory)
        parent_dir = "/home/iisajman/Desktop/Face-Recognition/Face_Images/"
        #path = os.path.join(parent_dir, directory)
        #os.mkdir(path)
        face_cascade= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        cap = cv2.VideoCapture(0)    
        count=0   
        while (True):
            ret, img = cap.read()
            #gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert Video frame to Greyscale
            faces = face_cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=5, minSize=(20,20)) #Recog. faces
            for (x, y, w, h) in faces:
                count+=1
                #roi_gray = gray[y:y+h, x:x+w] #Convert Face to greyscale
                #cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
#                #crop_face = img[y:y+h, x:x+w]
                #imageNp = cv2.resize(img, (200, 200), interpolation=cv2.INTER_AREA)
                #gray_img=cv2.cvtColor(imageNp,cv2.COLOR_BGR2GRAY) 
                data = Image.fromarray(img)
    #              equalized_image = cv2.equalizeHist(imageNp)
                os.chdir(parent_dir)            
                data.save(str(directory)+str(sid)+'.'+str(count)+".jpeg")        
            cv2.imshow('frame',img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
       
    win=tk.Tk()
    win.geometry("320x200")
    win.title("DATASET COLLECTION")
    L1=tk.Label(win,text="Enter the id:")
    L1.place(x=50,y=10)
    my_text_box=tk.Text(win, height=2, width=10)
    my_text_box.place(x=180,y=10)
    L2=tk.Label(win,text="Enter the name:")
    L2.place(x=50,y=50)
    my_text_box1=tk.Text(win, height=2, width=10)
    my_text_box1.place(x=180,y=50)
    Enter= tk.Button(win, height=3, width=5, text="Enter", command=lambda: get_input())
    Enter.place(x=180,y=100)
    Position= tk.Button(win, height=3, width=5, text="Position", command=lambda: face_tracking())
    Position.place(x=250,y=100)
    #sid=int(input("enter the id of the person"))
    #directory = input("enter the name of the folder:")    
    

def face_tracking():
    cap=cv2.VideoCapture(0)
    count=0
    while True:
        ret,frame=cap.read()
        count += 1
        if count % 10 != 0:
            continue
        frame=cv2.resize(frame,(640,480))
        #frame=cv2.flip(frame,-1)
        #obj_data(frame)
        image_input = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_face.process(image_input)
        if not results.detections:
           print("noface")
        else:    
             for detection in results.detections:
                 bbox = detection.location_data.relative_bounding_box
    #             print(bbox)
                 x, y, w, h = int(bbox.xmin*width), int(bbox.ymin * height), int(bbox.width*width),int(bbox.height*height)
                 cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
                 cx=int(x+x+w)//2
                 cy=int(y+y+h)//2
                 cv2.circle(frame,(cx,cy),5,(0,0,255),-1)
                 a=int(cx)//62
                 print("a",a)
                 pwm.start(a)               
        cv2.imshow("FRAME",frame)
        if cv2.waitKey(1)&0xFF==ord('q'):
            break
    cap.release()
    pwm.start(0)
    cv2.destroyAllWindows()
# def sensor1():
#     distance_to_object = sensor.distance * 100
#     print(distance_to_object)
#     if distance_to_object<=25:
#         motor2.off()
def move():
    newwin=tk.Toplevel()
    newwin.title("Robot Control")
    newwin.geometry('500x350')
    newwin.bind('<Escape>', lambda e: newwin.quit())
    def Brake():
        GPIO.output(Motor1A,GPIO.LOW)
        GPIO.output(Motor1B,GPIO.LOW)
        print("stop")
    def forward():
        GPIO.output(Motor1A,GPIO.HIGH)
        GPIO.output(Motor1B,GPIO.HIGH)
        print("going forward")
        #face_recog1()
        #count=0
    def sensor():
        while(True):
            distance_to_object = sensor.distance * 100
            print(distance_to_object)
            if distance_to_object<=25:
                GPIO.output(Motor1A,GPIO.LOW)
                GPIO.output(Motor1B,GPIO.LOW)
                call(["espeak","OBSTACLE DETECTED"])
            elif distance_to_object>25:
               GPIO.output(Motor1A,GPIO.HIGH)
               GPIO.output(Motor1B,GPIO.HIGH)
            
    B1=tk.Button(newwin,text="Brake",command=Brake,width=10,height=3)
    B1.place(x=200,y=150)
    B2=tk.Button(newwin,text="Fwd",command=forward,width=10,height=3)
    B2.place(x=200,y=50)
    B3=tk.Button(newwin,text="Obstacle",command=sensor,width=10,height=3)
    B3.place(x=200,y=350)
    #B4=tk.Button(newwin,text="left",command=left,width=10,height=3)
    #B4.place(x=50,y=150)
    #B5=tk.Button(newwin,text="right",command=right,width=10,height=3)
    #B5.place(x=350,y=150)
    B6=tk.Button(newwin,text="ShowCam",command=facedetection,width=10,height=3)
    B6.place(x=200,y=250)
    #button0 = PushButton(app, command=toggleSwitch, text="Start", width=10,height=3,grid=[2,4])
    #button1 = PushButton(app, command=forwardSpeedIncrease, text="Frwd Speed",width=10,height=3, grid=[2,3])
    #button2 = PushButton(app, command=backwardSpeedReduce, text="Bckwd Speed -",width=10,height=3, grid=[2,5])
    #button3 = PushButton(app, command=backwardSpeedIncrease, text = "Bckwd Speed+", width=10,height=3, grid=[1,4])
    #button4 = PushButton(app, command=forwardSpeedReduce, text="Frwd Speed -",width=10, height=3, grid=[3,4])
    #app.display()

def quitapp():
    top.destroy()
    
B1 = tk.Button(top,text ="Surveillance",fg='blue',command=face_recog1,width=30)
B1.place(x=380,y=300)
B2= tk.Button(top, text="Face Register",fg='blue',command=snapshot,width=30)
B2.place(x=380,y=350)
B3=tk.Button(top,text = 'Quit',fg='blue',command=quitapp,width=30)
B3.place(x=380,y=450)
#B4=tk.Button(top,text = 'Face Tracking',fg='blue',command=face_tracking,width=25)
#B4.place(x=100,y=300)
B5=tk.Button(top,text = 'Remote Control',fg='blue',command=move,width=30)
B5.place(x=380,y=400)
top.mainloop()
