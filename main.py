import numpy as np
from six.moves.urllib.request import urlopen
from six import BytesIO
from PIL  import Image,ImageOps
import tempfile
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf 
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse
import logging
from scipy.spatial import KDTree
from webcolors import hex_to_rgb, hex_to_rgb_percent, rgb_to_name ,CSS3_HEX_TO_NAMES
from sklearn.cluster import KMeans
import webcolors
import cv2
from collections import Counter

tf.get_logger().setLevel(logging.ERROR)

def create_model():
    model = tf.keras.applications.InceptionResNetV2(
    include_top=True,
    weights="imagenet",
    classes=1000,
    classifier_activation="softmax")
    return model


def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

def get_colors(image,number_of_colors,show_chart):
    modified_image = cv2.resize(image, (90, 90), interpolation = cv2.INTER_AREA)
    modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)
    clf = KMeans(n_clusters = number_of_colors ,n_init =20)
    labels = clf.fit_predict(modified_image)
    counts = Counter(labels)
    center_colors = clf.cluster_centers_
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]
  
    return rgb_colors

def convert_rgb_to_names(rgb_tuple):


    css3_db = webcolors.CSS3_HEX_TO_NAMES
    names = []
    rgb_values = []
    for color_hex, color_name in css3_db.items():
        names.append(color_name)
        rgb_values.append(hex_to_rgb(color_hex))
    
    kdt_db = KDTree(rgb_values)
    distance, index = kdt_db.query(rgb_tuple)
    return  names[index]


def predict_from_url(url):
    _ , filename = tempfile.mkstemp(suffix=".jpg")
    response = urlopen(url)
    image_data = response.read()
    image_data = BytesIO(image_data)
    pil_image = Image.open(image_data)
    pil_image = ImageOps.fit(pil_image, (299, 299), Image.ANTIALIAS)
    pil_image_rgb = pil_image.convert("RGB")
    x = image.img_to_array(pil_image_rgb)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    mod = create_model()
    preds = mod.predict(x)
    response =[(pred[1],str(pred[2])) for  pred in decode_predictions(preds ,top = 15)[0] ]

    return dict (response)  



def pred_color (url):
    _ , filename = tempfile.mkstemp(suffix=".jpg")
    response = urlopen(url)
    image_data = response.read()
    image_data = BytesIO(image_data) 
    pil_image = Image.open(image_data)
    imag= image.img_to_array(pil_image) 
    r = get_colors(imag,3, 1)
    l = []
    for i in range(len(r)):
        l.append(convert_rgb_to_names((r[i][0],(r[i][1]),( r[i][2]))))

    return l


app = FastAPI()
@app.get('/',response_class=HTMLResponse)
async def root():
    return '<h1> hello to Giza apps </h1>'


@app.post('/tags-colors/{file_path:path}')
async def predict_tags(file_path :str):
    create_model()
    res = predict_from_url(file_path)
    col = pred_color(file_path)


    return jsonable_encoder({"colors":col ,"tags": res })
 
