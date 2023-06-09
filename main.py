import base64
import os
import pyautogui
import cv2
from flask import Flask, request, send_from_directory

app = Flask(__name__)

camera = cv2.VideoCapture(0)


@app.route('/')
def hello():
    return send_from_directory('.', 'index.html')


@app.route('/save_image', methods=['POST'])
def save_image():
    image_data = request.json.get('image_data')
    filename = request.json.get('filename')
    save_path = os.path.join('images', filename)

    if not os.path.exists('images'):
        os.makedirs('images')

    with open(save_path, 'wb') as f:
        f.write(base64.b64decode(image_data.split(',')[1]))

    return 'Image saved'


@app.route('/capture_screenshot', methods=['POST'])
def capture_screenshot():
    filename = request.json.get('filename')
    save_path = os.path.join('camera', filename)

    if not os.path.exists('camera'):
        os.makedirs('camera')

    screenshot = pyautogui.screenshot()
    screenshot.save(save_path)

    return 'Screenshot captured'


if __name__ == '__main__':
    app.run()

