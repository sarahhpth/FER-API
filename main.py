from fastapi import FastAPI, File, UploadFile
import tensorflow as tf # use tf 2.16.1
from io import BytesIO
from PIL import Image
import numpy as np

app = FastAPI()

model = tf.keras.models.load_model('.\mobilenet.keras')

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    image = Image.open(BytesIO(await file.read()))
    image = image.resize((48, 48))
    image_array = np.array(image)/255.0
    image_array = np.expand_dims(image_array, axis = 0)

    # make predictions
    classes = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprised']

    prediction = model.predict(image_array)
    prediction_index = np.argmax(prediction)
    prediction_probability = np.max(prediction)
    prediction_class = classes[prediction_index]

    result = f'{prediction_class} {prediction_probability:.4f}'

    return {"result": result}