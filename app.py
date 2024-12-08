from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse, JSONResponse
import onnxruntime as ort
import numpy as np
import cv2
import io

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Mount static files
#app.mount("/", StaticFiles(directory=".", html=True), name="static")
app.mount("/static", StaticFiles(directory="./static", html=True), name="static")

# Mount the static directory
#app.mount("/static", StaticFiles(directory="./static"), name="static")

# Define the root endpoint
@app.get("/")
def read_root():
    return FileResponse("./static/index.html")


# Load your ONNX model
onnx_model_path = 'model.onnx'
session = ort.InferenceSession(onnx_model_path)

# Global variable to store latest prediction
current_prediction = {
    "label": "waiting",
    "confidence": 0.0
}
print(app.routes)


# Define the input name (assuming a single input)
input_name = session.get_inputs()[0].name

# Initialize OpenCV for video capture
video_capture = cv2.VideoCapture(0)

def generate_frames():
    global current_prediction
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
        
        # Update global prediction
        current_prediction = {
            "label": "spoof" if prediction > 0.5 else "real",
            "confidence": float(prediction)
        }
        
        # Label based on prediction
        label = current_prediction["label"]
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

@app.get("/current_prediction")
async def get_current_prediction():
    return JSONResponse(current_prediction)

# Cleanup function to release video capture when server stops
@app.on_event("shutdown")
def shutdown_event():
    video_capture.release()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)