from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import librosa
import io
import base64
from pydub import AudioSegment
import tempfile
import os
import time
import uuid
from emotion_model import VoiceEmotionDetector
import traceback

app = Flask(__name__)
CORS(app)

# Initialize the emotion detector
detector = VoiceEmotionDetector()
model_loaded = False

def load_model():
    """Load or train the emotion detection model"""
    global model_loaded
    try:
        if not model_loaded:
            if os.path.exists('models/emotion_model.h5'):
                print("Loading existing model...")
                success = detector.load_model()
                if success:
                    model_loaded = True
                    print("Model loaded successfully!")
                else:
                    print("Failed to load model. Training new model...")
                    detector.train_model()
                    model_loaded = True
            else:
                print("No pre-trained model found. Training new model...")
                detector.train_model()
                model_loaded = True
    except Exception as e:
        print(f"Error in load_model: {e}")
        traceback.print_exc()

# Load model when app starts
with app.app_context():
    load_model()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        return jsonify({
            'status': 'healthy',
            'model_loaded': model_loaded,
            'supported_emotions': list(detector.label_encoder.classes_) if detector.label_encoder else []
        })
    except Exception as e:
        return jsonify({'error': f'Health check failed: {str(e)}'}), 500

@app.route('/predict', methods=['POST'])
def predict_emotion():
    """Predict emotion from uploaded audio file"""
    try:
        if not model_loaded:
            return jsonify({'error': 'Model not loaded'}), 500
        
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        
        if audio_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Create unique filename to avoid conflicts
        unique_id = str(uuid.uuid4())
        temp_dir = tempfile.gettempdir()
        
        # Get original file extension
        original_filename = audio_file.filename
        file_extension = os.path.splitext(original_filename)[1] if original_filename else '.wav'
        if not file_extension:
            file_extension = '.wav'
        
        temp_filename = f"emotion_audio_{unique_id}{file_extension}"
        temp_path = os.path.join(temp_dir, temp_filename)
        
        try:
            # Save uploaded file with proper error handling
            audio_file.save(temp_path)
            
            # Verify file was saved
            if not os.path.exists(temp_path):
                return jsonify({'error': 'Failed to save uploaded file'}), 500
            
            # Check file size
            file_size = os.path.getsize(temp_path)
            if file_size == 0:
                return jsonify({'error': 'Uploaded file is empty'}), 400
            
            print(f"Processing file: {temp_path} (size: {file_size} bytes)")
            
            # Process the file
            result = detector.predict_emotion(temp_path)
            
            if result is None:
                return jsonify({'error': 'Failed to process audio - unable to extract features'}), 500
                
            return jsonify(result)
            
        except Exception as processing_error:
            print(f"Processing error: {processing_error}")
            traceback.print_exc()
            return jsonify({'error': f'Processing failed: {str(processing_error)}'}), 500
            
        finally:
            # Robust file cleanup
            cleanup_attempts = 0
            max_attempts = 5
            
            while cleanup_attempts < max_attempts and os.path.exists(temp_path):
                try:
                    # Force garbage collection to release file handles
                    import gc
                    gc.collect()
                    
                    # Wait a moment for any processes to release the file
                    time.sleep(0.1)
                    
                    # Attempt to delete
                    os.unlink(temp_path)
                    print(f"Successfully cleaned up {temp_path}")
                    break
                    
                except (OSError, PermissionError) as e:
                    cleanup_attempts += 1
                    if cleanup_attempts < max_attempts:
                        time.sleep(0.3 * cleanup_attempts)  # Progressive delay
                    else:
                        print(f"Warning: Could not delete {temp_path} after {max_attempts} attempts: {e}")
    
    except Exception as e:
        print(f"Unexpected error in predict: {e}")
        traceback.print_exc()
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/predict_stream', methods=['POST'])
def predict_emotion_stream():
    """Predict emotion from audio data stream (better for real-time)"""
    try:
        if not model_loaded:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get raw audio data
        audio_data = request.get_data()
        
        if len(audio_data) == 0:
            return jsonify({'error': 'No audio data received'}), 400
        
        print(f"Processing stream data: {len(audio_data)} bytes")
        
        # Process audio directly in memory (no file system)
        try:
            # Convert audio data to AudioSegment
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_data))
            
            # Convert to numpy array with proper data type handling
            if audio_segment.sample_width == 1:
                audio_array = np.array(audio_segment.get_array_of_samples(), dtype=np.int8)
            elif audio_segment.sample_width == 2:
                audio_array = np.array(audio_segment.get_array_of_samples(), dtype=np.int16)
            else:
                audio_array = np.array(audio_segment.get_array_of_samples(), dtype=np.int32)
            
            # Convert to float32 and normalize
            audio_array = audio_array.astype(np.float32)
            
            # Handle stereo audio by converting to mono
            if audio_segment.channels == 2:
                audio_array = audio_array.reshape((-1, 2)).mean(axis=1)
            
            # Normalize audio to [-1, 1] range
            if np.max(np.abs(audio_array)) > 0:
                audio_array = audio_array / np.max(np.abs(audio_array))
            
            # Check for minimum audio length
            min_samples = int(0.5 * audio_segment.frame_rate)  # At least 0.5 seconds
            if len(audio_array) < min_samples:
                return jsonify({'error': 'Audio too short - minimum 0.5 seconds required'}), 400
            
            print(f"Audio processed: {len(audio_array)} samples at {audio_segment.frame_rate}Hz")
            
            # Predict emotion directly from array
            result = detector.predict_emotion_from_array(audio_array, audio_segment.frame_rate)
            
            if result is None:
                return jsonify({'error': 'Failed to extract features from audio'}), 500
            
            return jsonify(result)
            
        except Exception as processing_error:
            print(f"Audio processing error: {processing_error}")
            traceback.print_exc()
            return jsonify({'error': f'Audio processing failed: {str(processing_error)}'}), 500
    
    except Exception as e:
        print(f"Stream processing error: {e}")
        traceback.print_exc()
        return jsonify({'error': f'Stream processing error: {str(e)}'}), 500

@app.route('/predict_file', methods=['POST'])
def predict_emotion_file():
    """Predict emotion from uploaded file using in-memory processing"""
    try:
        if not model_loaded:
            return jsonify({'error': 'Model not loaded'}), 500
        
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        
        if audio_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        try:
            # Read file directly into memory
            audio_data = audio_file.read()
            
            if len(audio_data) == 0:
                return jsonify({'error': 'Empty audio file'}), 400
            
            print(f"Processing uploaded file: {audio_file.filename} ({len(audio_data)} bytes)")
            
            # Reset file pointer (in case it's needed)
            audio_file.seek(0)
            
            # Process using AudioSegment (in-memory)
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_data))
            
            # Convert to numpy array with proper data type handling
            if audio_segment.sample_width == 1:
                audio_array = np.array(audio_segment.get_array_of_samples(), dtype=np.int8)
            elif audio_segment.sample_width == 2:
                audio_array = np.array(audio_segment.get_array_of_samples(), dtype=np.int16)
            else:
                audio_array = np.array(audio_segment.get_array_of_samples(), dtype=np.int32)
            
            # Convert to float32
            audio_array = audio_array.astype(np.float32)
            
            # Handle stereo to mono conversion
            if audio_segment.channels == 2:
                audio_array = audio_array.reshape((-1, 2)).mean(axis=1)
            
            # Normalize audio to [-1, 1] range
            if np.max(np.abs(audio_array)) > 0:
                audio_array = audio_array / np.max(np.abs(audio_array))
            
            # Check for minimum audio length
            min_samples = int(0.5 * audio_segment.frame_rate)  # At least 0.5 seconds
            if len(audio_array) < min_samples:
                return jsonify({'error': 'Audio file too short - minimum 0.5 seconds required'}), 400
            
            print(f"Audio processed: {len(audio_array)} samples at {audio_segment.frame_rate}Hz")
            
            # Predict emotion from array
            result = detector.predict_emotion_from_array(audio_array, audio_segment.frame_rate)
            
            if result is None:
                return jsonify({'error': 'Failed to extract features from audio file'}), 500
            
            return jsonify(result)
            
        except Exception as processing_error:
            print(f"File processing error: {processing_error}")
            traceback.print_exc()
            return jsonify({'error': f'File processing failed: {str(processing_error)}'}), 500
    
    except Exception as e:
        print(f"File upload error: {e}")
        traceback.print_exc()
        return jsonify({'error': f'File upload error: {str(e)}'}), 500

@app.route('/get_emotions', methods=['GET'])
def get_supported_emotions():
    """Get list of supported emotions"""
    try:
        if detector.label_encoder:
            return jsonify({
                'emotions': list(detector.label_encoder.classes_)
            })
        else:
            return jsonify({
                'emotions': ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
            })
    except Exception as e:
        print(f"Error getting emotions: {e}")
        return jsonify({'error': 'Failed to get supported emotions'}), 500

# Cleanup function for leftover temp files
def cleanup_temp_files():
    """Clean up any leftover temporary files"""
    try:
        temp_dir = tempfile.gettempdir()
        cleaned_count = 0
        
        for filename in os.listdir(temp_dir):
            if filename.startswith('emotion_audio_') and any(filename.endswith(ext) for ext in ['.wav', '.mp3', '.m4a', '.webm', '.ogg']):
                temp_path = os.path.join(temp_dir, filename)
                try:
                    os.unlink(temp_path)
                    cleaned_count += 1
                    print(f"Cleaned up leftover file: {filename}")
                except Exception as cleanup_error:
                    print(f"Could not clean up {filename}: {cleanup_error}")
        
        if cleaned_count > 0:
            print(f"Cleaned up {cleaned_count} leftover temp files")
            
    except Exception as e:
        print(f"Error during temp file cleanup: {e}")

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({'error': 'Method not allowed'}), 405

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    try:
        # Clean up any leftover temp files from previous runs
        print("Cleaning up temporary files...")
        cleanup_temp_files()
        
        # Create necessary directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('data', exist_ok=True)
        
        print("🎤 Voice Emotion Detection API Starting...")
        print("📡 API will be available at: http://localhost:5000")
        print("🔍 Health check: http://localhost:5000/health")
        print("📊 Supported endpoints:")
        print("  - POST /predict (file upload)")
        print("  - POST /predict_stream (raw audio data)")
        print("  - POST /predict_file (in-memory processing)")
        print("  - GET /get_emotions (list supported emotions)")
        
        if model_loaded:
            print("✅ Model loaded and ready!")
        else:
            print("⚠️ Model not loaded - will train on first request")
        
        app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        traceback.print_exc()
    finally:
        print("🧹 Final cleanup...")
        cleanup_temp_files()