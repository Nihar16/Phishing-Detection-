import os
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
import joblib
import pickle
import re
from flask_cors import CORS
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import imageio

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Upload folder for video files
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global variables for models
face_model = None
deepfake_model = None
email_model = None
sms_model = None
url_model = None

# Function to load models
def load_models():
    """
    Load face detection and deepfake detection models.
    This function attempts to load the models and sets the global variables.
    """
    global face_model, deepfake_model
    try:
        face_model_path = r'C:\\Users\\nihar\\OneDrive\\Desktop\\Phishing Detection software\\models\\face_detction_CNN_RNN.h5'
        deepfake_model_path = r'C:\\Users\\nihar\\OneDrive\\Desktop\\Phishing Detection software\\models\\CNN_RNN.h5'

        if not os.path.exists(face_model_path):
            raise FileNotFoundError(f"Face detection model not found at {face_model_path}")
        if not os.path.exists(deepfake_model_path):
            raise FileNotFoundError(f"Deepfake detection model not found at {deepfake_model_path}")

        face_model = load_model(face_model_path)
        print("Face detection model loaded successfully.")
        
        deepfake_model = load_model(deepfake_model_path)
        print("Deepfake detection model loaded successfully.")
        
        return {"status": "success", "message": "Models loaded successfully."}
    except Exception as e:
        print(f"Error loading models: {e}")
        return {"status": "error", "message": f"Error loading models: {e}"}

# Load pre-trained models and vectorizers
try:
    email_model = pickle.load(open('C:\\Users\\nihar\\OneDrive\\Desktop\\Phishing Detection software\\models\\email_spam_model.pkl', 'rb'))
    sms_model = joblib.load('C:\\Users\\nihar\\OneDrive\\Desktop\\Phishing Detection software\\models\\sms_spam_nb.pkl')
    url_model = joblib.load('C:\\Users\\nihar\\OneDrive\\Desktop\\Phishing Detection software\\models\\phishing.pkl')

    # Load vectorizers
    email_vectorizer = joblib.load(r'C:\\Users\\nihar\\OneDrive\\Desktop\\Phishing Detection software\\models\\email_vectorizer.pkl')
    sms_vectorizer = joblib.load(r'C:\\Users\\nihar\\OneDrive\\Desktop\\Phishing Detection software\\models\\sms_vectorizer.pkl')

    # Load deepfake models
    model_status = load_models()
    print(model_status["message"])

    print("Email, SMS, and URL models loaded successfully.")
except Exception as e:
    print(f"Error loading text-based models or vectorizers: {e}")

@app.route('/')
def index():
    """Serve the index.html file."""
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    """Serve favicon.ico as a static file."""
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/reload_models', methods=['POST'])
def reload_models():
    """
    API endpoint to reload models dynamically.
    """
    result = load_models()
    return jsonify(result)

@app.route('/analyze', methods=['POST'])
def analyze():
    """Handle requests for email, SMS, URL, or deepfake detection."""
    detection_type = request.form.get('detectionType')  # Get detection type
    input_text = request.form.get('inputText')  # Input text for analysis
    file = request.files.get('file')  # Uploaded file for video analysis

    try:
        if detection_type == 'email':
            return analyze_email(input_text)
        elif detection_type == 'sms':
            return analyze_sms(input_text)
        elif detection_type == 'url':
            return analyze_url(input_text)
        elif detection_type == 'deepfake' and file:
            filename = secure_filename(file.filename)
            if not filename.endswith('.mp4'):
                return jsonify({"status": "error", "message": "Invalid file format. Please upload an MP4 file."})

            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)  # Save the uploaded MP4 file

            # Call the deepfake detection function
            return analyze_video(file_path)

        return jsonify({"status": "error", "message": "Invalid request or no file uploaded."})
    except Exception as e:
        print(f"Error in analyze function: {e}")
        return jsonify({"status": "error", "message": "Failed to analyze input."})

def preprocess_text(text):
    """General text preprocessing function."""
    text = text.lower()
    text = re.sub(r'\W', ' ', text)  # Remove non-alphanumeric characters
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    return text

def analyze_email(email_text):
    """Analyze email text for phishing."""
    try:
        if not email_text.strip():
            return jsonify({"status": "error", "message": "Email text cannot be empty."})
        email_text = preprocess_text(email_text)
        feature_vector = email_vectorizer.transform([email_text]).toarray()
        prediction = email_model.predict(feature_vector)[0]
        return jsonify({"status": "success", "prediction": 'safe' if prediction == 0 else 'phishing'})
    except Exception as e:
        print(f"Error in analyze_email: {e}")
        return jsonify({"status": "error", "message": "An error occurred while analyzing email."})

def analyze_sms(sms_text):
    """Analyze SMS text for phishing."""
    try:
        sms_text = preprocess_text(sms_text)
        feature_vector = sms_vectorizer.transform([sms_text]).toarray()
        prediction = sms_model.predict(feature_vector)[0]
        return jsonify({"status": "success", "prediction": 'safe' if prediction == 0 else 'phishing'})
    except Exception as e:
        print(f"Error in analyze_sms: {e}")
        return jsonify({"status": "error", "message": "Failed to analyze SMS"})

def analyze_url(url_text):
    """Analyze URL for phishing."""
    try:
        prediction = url_model.predict([url_text.strip()])[0]
        status = 'safe' if prediction == 'good' else 'phishing'
        return jsonify({"status": "success", "prediction": status})
    except Exception as e:
        print(f"Error in analyze_url: {e}")
        return jsonify({"status": "error", "message": "Failed to analyze URL"})

def analyze_video(video_path):
    """
    Analyze an MP4 video for deepfake detection.
    - Extracts frames from the video.
    - Detects faces using a face detection model.
    - Analyzes frames with detected faces for deepfakes.
    """
    global face_model, deepfake_model

    # Check if models are loaded
    if face_model is None or deepfake_model is None:
        return jsonify({"status": "error", "message": "Models not loaded. Please check the model files."})

    try:
        print(f"Analyzing video: {video_path}")

        # Initialize video reader
        reader = imageio.get_reader(video_path, format="ffmpeg")
        frame_predictions = []
        frame_counter = 0

        for frame in reader:
            frame_counter += 1
            try:
                # Convert frame to PIL Image
                frame_image = Image.fromarray(frame)
                frame_image = frame_image.resize((128, 128))

                # Normalize frame data
                frame_array = np.array(frame_image).astype("float32") / 255.0
                frame_array = np.expand_dims(frame_array, axis=0)  # Add batch dimension

                # Face detection
                face_present = face_model.predict(frame_array)[0][0] > 0.5
                if face_present:
                    # Deepfake detection
                    prediction = deepfake_model.predict(frame_array)[0][0]
                    frame_predictions.append(prediction)
                else:
                    print(f"Frame {frame_counter}: No face detected.")
            except Exception as frame_error:
                print(f"Error processing frame {frame_counter}: {frame_error}")
                continue

        reader.close()

        if len(frame_predictions) == 0:
            return jsonify({"status": "success", "prediction": "No faces detected in the video."})

        # Aggregate predictions
        avg_prediction = np.mean(frame_predictions)
        result = "Deepfake detected" if avg_prediction > 0.5 else "Real video"

        return jsonify({"status": "success", "prediction": result})
    except Exception as e:
        print(f"Error in analyze_video: {e}")
        return jsonify({"status": "error", "message": "Failed to analyze video."})

if __name__ == '__main__':
    app.run(debug=True)
