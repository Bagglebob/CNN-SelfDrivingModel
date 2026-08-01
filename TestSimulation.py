import os
print('Setting Up ...')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import socketio
import eventlet
import numpy as np
from tensorflow.keras.models import load_model
from flask import Flask
import base64
from io import BytesIO
from PIL import Image
import cv2

from preprocess import preprocess


sio = socketio.Server()
app = Flask(__name__) #__main__
maxSpeed = 30

def preProcessing(img):
    # img = img[60:135, :, :]
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    # img = cv2.GaussianBlur(img, (3, 3), 0)
    # img = cv2.resize(img, (200, 66))
    # img = img / 255
    img = preprocess(img)
    return img


@sio.on('telemetry')
def telemetry(sid, data):
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = preProcessing(image)
    image = np.array([image])
    # steering = float(model.predict(image))
    # AI told me to use:
    # verbose=0 to suppress the progress bar
    # [0][0] to get the first element of the first element of the prediction
    # * 1.5 to make the steering more responsive, since it was trained on keyboard input
    steering = float(model.predict(image, verbose=0)[0][0])* 1.5
    throttle = 1.0 - speed/maxSpeed
    print(f'{throttle}, {steering}, {speed}')
    sendControl(steering, throttle)


@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    sendControl(0, 0)


def sendControl(steering, throttle):
    sio.emit('steer', data={
        'steering_angle' : steering.__str__(),
        'throttle' : throttle.__str__()
    })

if __name__ == "__main__":
    model = load_model('model.h5', compile=False)
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)