from flask import Flask, request
from cnn.yolo import YOLO
from PIL import Image
import numpy as np
import cv2
import json
app = Flask(__name__)
yolo = YOLO()

def detect_img(img):
  image = Image.fromarray(img)
  detections = yolo.detect_image(image)
  ret_det = dict()
  for key in detections:
    ret_det[yolo.GetClassFromIndex(key)] = int(detections[key])
  print(ret_det)
  return ret_det

@app.route('/', methods=['GET'])
def reach():
  return "CAN REACH!"

@app.route('/api/analyse', methods=['POST'])
def img_recog():
  print("Requesting obj detection")
  r = request
  # convert string of image data to uint8
  nparr = np.fromstring(r.data, np.uint8)
  # decode image
  img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
  return json.dumps(detect_img(img))
