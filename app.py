from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import onnxruntime as ort
import base64
from io import BytesIO
from PIL import Image
import cv2

# Initialize FastAPI app
app = FastAPI()

# Load the ONNX model
onnx_model_path = 'model.onnx'
session = ort.InferenceSession(onnx_model_path)

# Define the input name (assuming a single input)
input_name = session.get_inputs()[0].name

@app.post("/inference")
async def inference(frame: dict):
    try:
        # Decode the base64 image
        image_data = base64.b64decode(frame['image'].split(',')[1])  # Remove the data URL part
        image = Image.open(BytesIO(image_data))
        image = image.convert('RGB')
        
        # Preprocess the image for the model
        image = np.array(image)
        image = cv2.resize(image, (160, 160))  # Adjust to your model's input size
        image = image.astype('float32') / 255.0
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        
        # Perform inference with the ONNX model
        result = session.run(None, {input_name: image})
        prediction = result[0][0]

        # Make a prediction and return it
        label = "spoof" if prediction > 0.5 else "real"
        return JSONResponse({
            "label": label,
            "confidence": float(prediction)
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Start the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
