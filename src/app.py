from flask import Flask, Response, request
from cnn.yolo import YOLO
from PIL import Image
import tensorflow as tf
from io import BytesIO
import numpy as np
import cv2
import json
app = Flask(__name__)
LATEST_IMG = None

def detect_img(img):
  global LATEST_IMG
  ret_det = dict()
  with graph.as_default():
    image = Image.fromarray(img)
    detections, out = yolo.detect_image(image)
    for key in detections:
      ret_det[yolo.GetClassFromIndex(key)] = int(detections[key])
    print(ret_det)
    with BytesIO() as output:
      image.save(output, 'jpeg')
      LATEST_IMG = output.getvalue()
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
           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
  return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route('/api/analyse', methods=['POST'])
def img_recog():
  r = request
  # convert string of image data to uint8
  nparr = np.fromstring(r.data, np.uint8)
  # decode image
  img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
  return json.dumps(detect_img(img))

if __name__ == "__main__":
  global yolo
  yolo = YOLO()
  global graph
  graph = tf.get_default_graph()
  app.run(host='0.0.0.0')


