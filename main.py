import base64
import os
from PIL import Image
import uuid
from predictor import Predictor
from flask import Flask, request, send_from_directory, jsonify

app = Flask(__name__)

predictor = Predictor('./models/model_v4')


@app.route('/')
def hello():
    return send_from_directory('.', 'index.html')


@app.route('/save_image', methods=['POST'])
def save_image():
    image_data = request.json.get('image_data')
    unique_filename = str(uuid.uuid4()) + ".jpeg"
    save_path = os.path.join("images", unique_filename)

    if not os.path.exists('images'):
        os.makedirs('images')

    with open(save_path, 'wb') as f:
        f.write(base64.b64decode(image_data.split(',')[1]))

    img = Image.open(save_path)

    result, transformed_img = predictor(img)

    # Save the transformed image
    transformed_img_path = os.path.join("trans_img", "transformed_" + unique_filename)
    transformed_img.save(transformed_img_path)

    # Convert the transformed image to base64 for sending to frontend
    with open(transformed_img_path, "rb") as f:
        transformed_img_data = f.read()
        transformed_img_base64 = base64.b64encode(transformed_img_data).decode("utf-8")

    return jsonify(result=result, transformed_img=transformed_img_base64)


@app.route('/save-photo', methods=['POST'])
def save_photo():
    image_data_url = request.form['image_data_url']
    unique_filename = str(uuid.uuid4()) + ".jpeg"
    save_path = os.path.join("camera", unique_filename)

    # Decode and save the image data
    with open(save_path, "wb") as f:
        f.write(base64.b64decode(image_data_url.split(',')[1]))

    img = Image.open(save_path)

    result, transformed_img = predictor(img)

    # Save the transformed image
    transformed_img_path = os.path.join("trans_cam", "transformed_" + unique_filename)
    transformed_img.save(transformed_img_path)

    # Convert the transformed image to base64 for sending to frontend
    with open(transformed_img_path, "rb") as f:
        transformed_img_data = f.read()
        transformed_img_base64 = base64.b64encode(transformed_img_data).decode("utf-8")

    return jsonify(result=result, transformed_img=transformed_img_base64)


if __name__ == '__main__':
    app.run()

