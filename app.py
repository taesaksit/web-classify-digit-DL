from flask import Flask , render_template , request
from tensorflow.image import rgb_to_grayscale
from tensorflow.keras.preprocessing.image import load_img , img_to_array
from PIL import Image ,  ImageOps
from tensorflow.keras.models import load_model

from flask_cors import CORS

import numpy as np
import os
import pickle




app = Flask(__name__)
CORS(app)



@app.route('/')
def index():
    return render_template('index.html')

def process_image(img_name):
    img = load_img(img_name, target_size=(28, 28))
    img_ops = ImageOps.invert(img)
    img_arr = img_to_array(img_ops)
    img_gray = rgb_to_grayscale(img_arr)
    img_tensor = np.expand_dims(img_gray, axis=0) / 255


    return  predict_digit(img_tensor)

def predict_digit(img_tensor):
    model = load_model('./static/model_file/modelDigi.h5')

    y_pred = model.predict(img_tensor)
    return np.argmax(y_pred , axis=1)


@app.route('/upload_file' , methods=['POST'])
def upload_file():
    img_file = request.files['file_image_digit']
    folder = './static/uploads/'
    img_path = os.path.join(folder , img_file.filename)
    img_file.save(img_path)

    predicted = process_image(img_path)

    return render_template('index.html' , result=predicted , image_predicted=img_path )


if __name__ =='__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=5001, debug=True)

