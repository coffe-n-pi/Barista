from flask import Flask, Response, request
from cnn.yolo import YOLO
from PIL import Image
import numpy as np
import cv2
import json
app = Flask(__name__)
#yolo = YOLO()
LATEST_IMG = None

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

def gen():
  global LATEST_IMG
  """Video streaming generator function."""
  while True:
    frame = LATEST_IMG
    yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
  """Video streaming route. Put this in the src attribute of an img tag."""
  return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route('/api/analyse', methods=['POST'])
def img_recog():
  global LATEST_IMG
  print("Requesting obj detection")
  r = request
  # convert string of image data to uint8
  nparr = np.fromstring(r.data, np.uint8)
  # decode image
  img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
  LATEST_IMG = r.data
  return "{}" #json.dumps(detect_img(img))
