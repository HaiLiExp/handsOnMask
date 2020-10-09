# handsOnMask
The purpose of this project is to develop a code to recognize the action that a person touches his facemask.

Version V0.1
This version can only detect faces and it utilize the face detection model impletemented in OpenCV which is based on haar features. It is deprecated due to the high rate of false positive detections.

version V0.2
This version is based on yolo3. It can detect both faces and hands a mask. The faces are classifed to two classes: with and without a mask.

The face detection model is based on
https://github.com/sthanhng/yoloface
The hand detection model is based on 
https://github.com/cansik/yolo-hand-detection

FaceMask.ipynb
Based on MobileNetV2, this code is to classify a face image to two classes: with and without a mask. The model is saved as mask_recog.h5.
This code is based on https://www.mygreatlearning.com/blog/real-time-face-detection/

handsOnFace.py
This code is currently tested for webcam online detection and for local computation.
