from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form, Request
from fastapi.responses import JSONResponse
from fastapi.exception_handlers import http_exception_handler
from fastapi.exceptions import RequestValidationError
import tensorflow as tf # use tf 2.16.1 to load .keras model. older versions require .h5
from io import BytesIO
from PIL import Image
import numpy as np
from pydantic import BaseModel
import cv2
import base64

class PredictionRequest(BaseModel):
    class_name: str

app = FastAPI()

model = tf.keras.models.load_model('./mobilenet.keras')
classes = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprised']

# haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def crop_face(image):
    # convert PIL image to np array
    image_np = np.array(image)
    # grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    # detect face in image
    faces = face_cascade.detectMultiScale(gray, 1.2, 4)

    if len(faces) == 0:
        raise HTTPException(status_code=400, detail="No face detected")

    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        # to maintain the 3 channels, convert back to rgb but still in grayscale
        face_roi_rgbgray = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)

    final_image_cv2 = cv2.resize(face_roi_rgbgray, (48, 48)) # for base64
    final_image = np.expand_dims(final_image_cv2, axis = 0)
    final_image = final_image/255.0

    return final_image, final_image_cv2

def encode_image_to_base64(image):
    buffered = BytesIO()
    image = Image.fromarray(image.astype(np.uint8))  # Ensure image is uint8
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    print(f"HTTP Exception: {exc.detail}")
    return await http_exception_handler(request, exc)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    print(f"Validation Error: {exc}")
    return JSONResponse(
        status_code=400,
        content={"detail": exc.errors()},
    )

@app.get("/", status_code=200)
def health_check():
    return {"status": "healthy"}

@app.post('/predict', status_code=200)
async def predict(file: UploadFile = File(...), 
                  class_name: str = Form(...),
                  ):
    image = Image.open(BytesIO(await file.read()))
    face_image, face_image_cv2 = crop_face(image)

    # make predictions
    prediction = model.predict(face_image)
    prediction_probabilities = prediction[0].tolist()

    if class_name in classes:
        class_index = classes.index(class_name)
        prediction_probability = prediction[0][class_index]
        encoded_face_image = encode_image_to_base64(face_image_cv2)

        response = {
            "result": f'{class_name} {prediction_probability:.4f}',
            "result_accuracy": f'{prediction_probability * 100:.0f}%',
            "most_probable_class": classes[np.argmax(prediction)],
            "all_class_accuracy": [{ "class": cls, "accuracy": f'{prob * 100:.2f}%' } for cls, prob in zip(classes, prediction_probabilities)],
            "face_image": encoded_face_image,
            "message": "Prediction completed successfully"
        }
        return JSONResponse(content=response)
    else: 
        raise HTTPException(status_code=400, detail="Class not found")
    

# to run: 
# uvicorn main:app --reload