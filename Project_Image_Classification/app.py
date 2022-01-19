from __future__ import division,print_function
import numpy as np

import sys
import os
import glob
import re

from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

# Flask utils
from flask import Flask,redirect,url_for,request,render_template

# define a flask app
app = Flask(__name__)
MODEL_PATH = 'vgg19.tf'

# Load model
model = load_model(MODEL_PATH)
# model._make_predict_function() # Necessary
model.make_predict_function()

# Preprocessing function
def model_predict(img_path, model):
    img = image.load_img(img_path,target_size=(224,224))
    # Preprocessing the image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    return preds

@app.route('/',methods=['GET'])
def index():
    return render_template('index.html') # responsible for showing first page

@app.route('/predict',methods = ['GET','POST'])
def uplaod():
    if request.method == "POST":
        # Get the file from the post
        f = request.files['file']
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath,'uploads',secure_filename(f.filename))
        f.save(file_path)
        # here we make prediction
        pred = model_predict(file_path,model)
        pred_class = decode_predictions(pred,top=1) #Imagenet Decode
        result = str(pred_class[0][0][1]) # Convert to string
        return result
    return None

if __name__ == '__main__':
    app.run(debug=True)