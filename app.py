import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import tensorflow as tf

IMG_WIDTH = 64
IMG_HEIGHT = 64

UPLOAD_FOLDER = '/path/to/the/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

MODEL = tf.keras.models.load_model("frankie.h5")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html")

@app.route('/upload', methods=["POST"])
def upload():
    if request.method == "POST":
        print(request.files)
        if 'file' not in request.files:
            return redirect(url_for('index'))

    image = request.files["file"]
    if image and allowed_file(image.filename):
        img = cv2.imdecode(np.fromstring(request.files['file'].read(), np.uint8), cv2.IMREAD_UNCHANGED)
        resized_img = cv2.resize(img, dsize=(IMG_WIDTH, IMG_HEIGHT))
        print(resized_img)
        prediction = MODEL.predict([np.array(resized_img).reshape(1, IMG_WIDTH, IMG_HEIGHT, 3)])
        frankie = True if prediction[0] > .5 else False
        return render_template("results.html", frankie=frankie)
    else:
        return redirect(url_for('index'))




if __name__ == "__main__":
    app.run(debug=True)
