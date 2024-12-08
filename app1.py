from fastapi import FastAPI, Response
import onnxruntime as ort
import numpy as np
import cv2
from fastapi.responses import StreamingResponse
import io

# Initialize FastAPI app
app = FastAPI()

# Load your ONNX model
onnx_model_path = 'model.onnx'
session = ort.InferenceSession(onnx_model_path)

# Define the input name (assuming a single input)
input_name = session.get_inputs()[0].name

# Initialize OpenCV for video capture
video_capture = cv2.VideoCapture(0)

def generate_frames():
    while True:
        # Capture frame from the webcam
        ret, frame = video_capture.read()
        if not ret:
            break
        
        # Pre-process the image for the model
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (160, 160))  # Adjust to your model's input size
        image = image.astype('float32') / 255.0
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        
        # Perform inference with the ONNX model
        result = session.run(None, {input_name: image})
        prediction = result[0][0]
        
        # Label based on prediction
        label = "spoof" if prediction > 0.5 else "real"
        color = (0, 0, 255) if label == "spoof" else (0, 255, 0)

        # Draw bounding box and label
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Convert frame to byte format to stream it
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = io.BytesIO(buffer)
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes.read() + b'\r\n')

@app.get("/video_feed")
async def video_feed():
    # Stream the video feed in a MJPEG format
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

