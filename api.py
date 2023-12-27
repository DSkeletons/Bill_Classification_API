import flask
from flask import Flask, request, jsonify
import tensorflow as tf
import cv2
import numpy as np

app = Flask(__name__)

model = tf.keras.models.load_model('./model/bill_classification_model.keras')


def image_preprocessing(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 144))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


@app.route('/', methods=['GET', 'POST'])
def home():
    return flask.render_template('./front/index.html')

@app.route('/classification/predict', methods=['POST'])
def predict_image():
    try:

        request_img = request.files['image']

        if(request_img == NULL):
            return jsonify({'error': "You didn't provide an image."})

        temp = 'prediction_image.jpg'
        request_img.save(temp)
        img = image_preprocessing(temp)
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction)
        class_labels = [
            '1 EGP',
            '5 EGP',
            '10 EGP',
            'New 10 EGP',
            '20 EGP',
            'New 20 EGP',
            '50 EGP',
            '100 EGP',
            '200 EGP'
        ]
        predicted_label = class_labels[predicted_class]

        return jsonify({'prediction': predicted_label})

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run()
