FROM python:3.6.8-stretch

COPY Pipfile Pipfile.lock /
COPY ./src /src
COPY KERAS /KERAS
ADD https://pjreddie.com/media/files/yolov3.weights /KERAS

RUN pip3 install pipenv && \
    pipenv install

WORKDIR /KERAS
RUN pipenv run python convert.py yolov3.cfg yolov3.weights ../src/cnn/model_data/yolo.h5

WORKDIR /src
CMD ["pipenv", "run", "flask", "run", "--without-threads", "--host=0.0.0.0"]