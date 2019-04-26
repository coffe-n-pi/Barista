import requests
import json
import cv2
vid = cv2.VideoCapture(0)
video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
video_fps       = vid.get(cv2.CAP_PROP_FPS)
video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))

addr = 'http://localhost:5000'
test_url = addr + '/api/test'

# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}
ret, frame = vid.read()
# encode image as jpeg
_, img_encoded = cv2.imencode('.jpg', frame)
# send http request with image and receive response
response = requests.post(test_url, data=img_encoded.tostring(), headers=headers)
# decode response
print(response.text)
