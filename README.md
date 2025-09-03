# Real-voice-emotion-detection-app

A real-time voice emotion analysis web application that uses deep learning to detect emotions from audio input. Built with Python Flask backend and vanilla JavaScript frontend.

## Features

- **Real-time Emotion Detection**: Analyze emotions from live microphone input or uploaded audio files
- **8 Emotion Categories**: Detects neutral, calm, happy, sad, angry, fearful, disgust, and surprised emotions
- **Confidence Scoring**: Shows model confidence levels for each emotion prediction
- **Multiple Input Methods**: 
  - Live microphone recording with audio visualization
  - File upload support (WAV, MP3, M4A, WebM, OGG)
- **Responsive Web Interface**: Clean, modern UI that works on desktop and mobile
- **Demo Mode**: Fallback simulation when API is unavailable

## Tech Stack

### Backend
- **Python 3.7+**
- **Flask** - Web framework
- **TensorFlow/Keras** - Deep learning model
- **Librosa** - Audio feature extraction
- **NumPy & Pandas** - Data processing
- **Scikit-learn** - Machine learning utilities

### Frontend
- **HTML5** - Structure and audio handling
- **CSS3** - Responsive styling with animations
- **Vanilla JavaScript** - Interactive UI and API communication

### Machine Learning
- **Dataset**: RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
- **Features**: MFCC, Spectral Centroid, Spectral Rolloff, Zero Crossing Rate, Chroma Features, Mel Spectrogram
- **Model**: Deep Neural Network with dropout and batch normalization

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager
- Modern web browser with microphone support

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/voice-emotion-detection.git
cd voice-emotion-detection
```

2. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

3. **Create required directories**
```bash
mkdir models data
```

4. **Run the application**
```bash
python app.py
```

5. **Open your browser**
Navigate to `http://localhost:5000`

## Usage

### Recording Audio
1. Click "Start Recording" button
2. Allow microphone permissions when prompted
3. Speak into your microphone
4. Click "Stop Recording" to analyze

### Uploading Files
1. Click "Upload Audio File" button
2. Select an audio file (WAV, MP3, M4A, WebM, OGG)
3. Wait for analysis results

### Understanding Results
- **Primary Emotion**: The detected emotion with highest confidence
- **Confidence Level**: How certain the model is about the prediction (0-100%)
- **Emotion Bars**: Visual breakdown of all emotion probabilities

## API Endpoints

### Health Check
```
GET /health
```
Returns API status and model information.

### Predict from File Upload
```
POST /predict_file
Content-Type: multipart/form-data
Body: audio file
```

### Predict from Audio Stream
```
POST /predict_stream
Content-Type: audio/webm
Body: raw audio data
```

### Get Supported Emotions
```
GET /get_emotions
```
Returns list of supported emotion categories.

## Model Architecture

The emotion detection model uses:
- **Input Layer**: 64 audio features
- **Hidden Layers**: Dense layers with ReLU activation, batch normalization, and dropout
- **Output Layer**: 8 neurons with softmax activation for emotion probabilities
- **Training**: Uses synthetic data generation for demonstration purposes

### Audio Feature Extraction
- **MFCC**: 13 Mel-frequency cepstral coefficients (mean + std)
- **Spectral Features**: Centroid and rolloff statistics
- **Zero Crossing Rate**: Mean and standard deviation
- **Chroma Features**: 12 pitch class features
- **Mel Spectrogram**: 20 mel-scale frequency bands

## File Structure

```
voice-emotion-detection/
├── app.py                 # Flask web server
├── emotion_model.py       # ML model and training
├── index.html            # Frontend interface
├── models/               # Trained model files
├── data/                 # Dataset and temporary files
└── requirements.txt      # Python dependencies
```

## Configuration

### Environment Variables
- `FLASK_ENV`: Set to 'development' for debug mode
- `PORT`: Server port (default: 5000)

### Model Parameters
- Sample rate: 22050 Hz
- Audio duration: 3 seconds max per analysis
- Feature vector size: 64 dimensions

## Browser Compatibility

- **Chrome 60+**: Full support
- **Firefox 55+**: Full support
- **Safari 11+**: Full support
- **Edge 79+**: Full support

**Note**: Microphone access requires HTTPS in production environments.

## Limitations

- Audio files larger than 100MB are rejected
- Minimum audio duration: 0.5 seconds
- Real-time processing depends on system performance
- Model trained on synthetic data for demonstration purposes

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **RAVDESS Dataset**: Ryerson Audio-Visual Database of Emotional Speech and Song
- **Librosa Library**: For comprehensive audio analysis tools
- **TensorFlow Team**: For the deep learning framework
- **Flask Community**: For the lightweight web framework

## Contact

Your Name - [@yourusername](https://twitter.com/yourusername) - email@example.com

Project Link: [https://github.com/yourusername/voice-emotion-detection](https://github.com/yourusername/voice-emotion-detection)

---

**⭐ Star this repository if you found it helpful!**
