# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 09:04:07 2020

@author: Administrator
"""

import argparse
import sys
import os
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model


##############################################################################

from yoloFaceMaster.utils import *

faceParser = argparse.ArgumentParser()
faceParser.add_argument('--faceModel_cfg', type=str, 
                        default='yoloFaceMaster/cfg/yolov3-face.cfg',
                        help='path to config file of face detection model')
faceParser.add_argument('--faceModel_weights', type=str,
                        default='yoloFaceMaster/model-weights/yolov3-wider_16000.weights',
                        help='path to weights of face detection model')
faceArgs = faceParser.parse_args()

##############################################################################

import time
from yoloHandDetectionMaster.yolo import YOLO

handParser = argparse.ArgumentParser()
handParser.add_argument('-n', '--network', 
                        default="normal", 
                        help='Network Type: normal / tiny / prn')
handParser.add_argument('-d', '--device',
                        default=0, 
                        help='Device to use')
handParser.add_argument('-s', '--size', 
                        default=416, 
                        help='Size for yolo')
handParser.add_argument('-c', '--confidence',
                        default=0.2, 
                        help='Confidence for yolo')
handArgs = handParser.parse_args()

##############################################################################

dataParser = argparse.ArgumentParser()
dataParser.add_argument('--image', type=str, 
                        default='',
                        help='path to image file')
dataParser.add_argument('--video', type=str, 
                        default='',
                        help='path to video file')
dataParser.add_argument('--src', type=int,
                        default=0,
                        help='source of the camera')
dataParser.add_argument('--output-dir', type=str,
                        default='outputs/',
                        help='path to the output directory')
dataArgs = dataParser.parse_args()

##############################################################################

print('----- info -----')
print('[i] face detection config: ', faceArgs.faceModel_cfg)
print('[i] face detection model weights: ', faceArgs.faceModel_weights)
print('[i] Path to image file: ', dataArgs.image)
print('[i] Path to video file: ', dataArgs.video)
print('###########################################################\n')

##############################################################################

# check outputs directory
if not os.path.exists(dataArgs.output_dir):
    print('==> Creating the {} directory...'.format(dataArgs.output_dir))
    os.makedirs(dataArgs.output_dir)
else:
    print('==> Skipping create the {} directory...'.format(dataArgs.output_dir))
    
##############################################################################

# Give the configuration and weight files for the model and load the network
# using them.
facenet = cv2.dnn.readNetFromDarknet(faceArgs.faceModel_cfg, faceArgs.faceModel_weights)
facenet.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
facenet.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def interArea(boxA,boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
    # compute the area of intersection rectangle
    intersection = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    return intersection


def _main():
    window_name = 'hands on facemask detection using YOLOv3'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

##############################################################################
    output_file = ''

    if dataArgs.image:
        if not os.path.isfile(dataArgs.image):
            print("[!] ==> Input image file {} doesn't exist".format(dataArgs.image))
            sys.exit(1)
        cap = cv2.VideoCapture(dataArgs.image)
        output_file = dataArgs.image[:-4].rsplit('/')[-1] + '_handsOnFace.jpg'
    elif dataArgs.video:
        if not os.path.isfile(dataArgs.video):
            print("[!] ==> Input video file {} doesn't exist".format(dataArgs.video))
            sys.exit(1)
        cap = cv2.VideoCapture(dataArgs.video)
        output_file = dataArgs.video[:-4].rsplit('/')[-1] + '_handsOnFace.avi'
    else:
        # Get data from the camera
        cap = cv2.VideoCapture(dataArgs.src)

    # Get the video writer initialized to save the output video
    if not dataArgs.image:
        video_writer = cv2.VideoWriter(os.path.join(dataArgs.output_dir, output_file),
                                       cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                       cap.get(cv2.CAP_PROP_FPS), (
                                           round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                           round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
##############################################################################

    if handArgs.network == "normal":
        print("loading yolo...")
        handYolo = YOLO("yoloHandDetectionMaster/models/cross-hands.cfg", 
                        "yoloHandDetectionMaster/models/cross-hands.weights",
                        ["hand"])
    elif handArgs.network == "prn":
        print("loading yolo-tiny-prn...")
        handYolo = YOLO("yoloHandDetectionMaster/models/cross-hands-tiny-prn.cfg",
                        "yoloHandDetectionMaster/models/cross-hands-tiny-prn.weights", 
                        ["hand"])
    else:
        print("loading yolo-tiny...")
        handYolo = YOLO("yoloHandDetectionMaster/models/cross-hands-tiny.cfg",
                        "yoloHandDetectionMaster/models/cross-hands-tiny.weights",
                        ["hand"])

    handYolo.size = int(handArgs.size)
    handYolo.confidence = float(handArgs.confidence)
##############################################################################
    maskModel = load_model("mask_recog.h5")

    while True:
        
        time.sleep(4)

        has_frame, frame = cap.read()

        # Stop the program if reached end of video
        if not has_frame:
            print('[i] ==> Done processing!!!')
            print('[i] ==> Output file is stored at', os.path.join(dataArgs.output_dir, output_file))
            cv2.waitKey(1000)
            break

##############################################################################
        # Create a 4D blob from a frame.
        faceBlob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),
                                     [0, 0, 0], 1, crop=False)

        # Sets the input to the network
        facenet.setInput(faceBlob)

        # Runs the forward pass to get output of the output layers
        faceOuts = facenet.forward(get_outputs_names(facenet))

        # Remove the bounding boxes with low confidence
        faces = post_process(frame, faceOuts, CONF_THRESHOLD, NMS_THRESHOLD)
        print('[i] ==> # detected faces: {}'.format(len(faces)))
        print('#' * 60)

        # initialize the set of information we'll displaying on the frame
        info = [
            ('number of faces detected', '{}'.format(len(faces)))
        ]

        for (i, (txt, val)) in enumerate(info):
            text = '{}: {}'.format(txt, val)
            cv2.putText(frame, text, (10, (i * 20) + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)
##############################################################################
        faces_list=[]
        for (x, y, w, h) in faces:
                face_frame = frame[y:y+h,x:x+w]
                face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
                face_frame = cv2.resize(face_frame, (224, 224))
                face_frame = img_to_array(face_frame)
                face_frame = np.expand_dims(face_frame, axis=0)
                face_frame =  preprocess_input(face_frame)
                faces_list.append(face_frame)
                if len(faces_list)>0:
                    preds = maskModel.predict(faces_list)
                for pred in preds:
                    (mask, withoutMask) = pred
                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
                cv2.putText(frame, label, (x, y- 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            
                cv2.rectangle(frame, (x, y), (x + w, y + h),color, 2)
        
##############################################################################
        hands = []
        width, height, inference_time, results = handYolo.inference(frame)
        for detection in results:
            id, name, confidence, x, y, w, h = detection
            cx = x + (w / 2)
            cy = y + (h / 2)
            hands.append([x,y,w,h])
            # draw a bounding box rectangle and label on the image
            color = (0, 255, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "%s (%s)" % (name, round(confidence, 2))
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)
        for face in faces:
            for hand in hands:
                intersection = interArea(face, hand)
                if intersection > 0:
                     text = 'hands on facemask'
                     print("intersection ={}".format(intersection)) 
                     cv2.putText(frame, text, (face[0], face[1] - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)
                     break;
                     
##############################################################################

        # Save the output video to file
        if dataArgs.image:
            cv2.imwrite(os.path.join(dataArgs.output_dir, output_file), frame.astype(np.uint8))
        else:
            video_writer.write(frame.astype(np.uint8))

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            print('[i] ==> Interrupted by user!')
            break

##############################################################################

    cap.release()
    cv2.destroyAllWindows()

    print('==> All done!')
    print('***********************************************************')


if __name__ == '__main__':
    _main()
