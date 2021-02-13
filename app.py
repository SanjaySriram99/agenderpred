import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from flask import Flask, redirect, url_for, request, render_template

# define a Flask app
app = Flask(__name__)
# MODEL_VGG16 = load_model('C:/model3.h5')
graph = tf.compat.v1.get_default_graph()

print('Successfully loaded model...')
print('Visit http://127.0.0.1:5000')

def model_predict(img_path):
    '''
        helper method to process an uploaded image
    '''
    image = load_img(img_path, target_size=(200, 200))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)

    global graph
    with graph.as_default():
        MODEL = load_model('C:/AzureProject/model4.h5')
        preds = MODEL.predict(image)
    
    return preds

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # get the file from the HTTP-POST request
        f = request.files['file']        
        
        # save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', f.filename)
        f.save(file_path)
        
        # make prediction about this image's class
        preds = model_predict(file_path)
        
        age = int(preds[0][0].round())
        gender = int(preds[1][0].round())
        gend = None
        age_group = None
        if gender == 1:
            gend = 'Female'
            if age > 18:
                age_group = 'Adult'
            else:
                age_group = 'Child'
        elif gender == 0:
            gend = 'Male'
            if age > 18:
                age_group = 'Adult'
            else:
                age_group = 'Child'
        
        result = "Age: "+ str(age) + " Class: " + gend + " " + age_group 
        
        return result
    
    return None


if __name__ == '__main__':
    app.run(port=5000, debug=True)