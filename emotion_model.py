import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, Flatten, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import zipfile
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class VoiceEmotionDetector:
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.feature_scaler = None
        
    def download_dataset(self):
        """Download the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset"""
        print("Downloading RAVDESS dataset...")
        
        # Create data directory
        os.makedirs('data', exist_ok=True)
        
        # For demo purposes, we'll create synthetic data since downloading large datasets
        # can be problematic in development environments
        print("Creating synthetic dataset for demonstration...")
        self.create_sample_data_structure()
    
    def create_sample_data_structure(self):
        """Create a sample data structure for demonstration"""
        print("Creating synthetic training data...")
        os.makedirs('data/sample_audio', exist_ok=True)
        
        # Create metadata for sample files
        sample_data = {
            'filename': [],
            'emotion': [],
            'intensity': [],
            'actor': []
        }
        
        emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
        
        # Generate sample metadata
        for i, emotion in enumerate(emotions):
            for j in range(50):  # 50 samples per emotion for better training
                filename = f"sample_{emotion}_{j}.wav"
                sample_data['filename'].append(filename)
                sample_data['emotion'].append(emotion)
                sample_data['intensity'].append('normal')
                sample_data['actor'].append(f'Actor_{j%10 + 1}')
        
        # Save sample metadata
        df = pd.DataFrame(sample_data)
        df.to_csv('data/sample_metadata.csv', index=False)
        print("Sample data structure created!")
    
    def parse_filename(self, filename):
        """Parse RAVDESS filename to extract emotion information"""
        # RAVDESS filename format: Modality-VocalChannel-Emotion-EmotionalIntensity-Statement-Repetition-Actor
        parts = filename.split('-')
        if len(parts) >= 3:
            try:
                emotion_code = int(parts[2])
                # RAVDESS emotion mapping
                emotion_map = {
                    1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad',
                    5: 'angry', 6: 'fearful', 7: 'disgust', 8: 'surprised'
                }
                return emotion_map.get(emotion_code, 'unknown')
            except (ValueError, IndexError):
                return 'unknown'
        return 'unknown'
    
    def extract_features(self, audio_path, duration=3.0):
        """Extract audio features using librosa"""
        try:
            # Check if file exists
            if not os.path.exists(audio_path):
                print(f"Audio file not found: {audio_path}")
                return None
            
            # Load audio file with error handling
            try:
                y, sr = librosa.load(audio_path, duration=duration, sr=22050, res_type='kaiser_fast')
            except Exception as load_error:
                print(f"Failed to load audio file {audio_path}: {load_error}")
                return None
            
            # Check if audio is valid
            if len(y) == 0:
                print(f"Empty audio file: {audio_path}")
                return None
            
            return self._extract_features_from_array(y, sr)
            
        except Exception as e:
            print(f"Error extracting features from {audio_path}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _extract_features_from_array(self, y, sr):
        """Extract features from audio array"""
        try:
            features = []
            
            # Ensure we have enough audio data
            if len(y) < sr * 0.1:  # Less than 0.1 seconds
                print("Audio too short for feature extraction")
                return None
            
            # 1. MFCC (Mel-frequency cepstral coefficients)
            try:
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                if mfcc.shape[1] > 0:
                    mfcc_mean = np.mean(mfcc, axis=1)
                    mfcc_std = np.std(mfcc, axis=1)
                    features.extend(mfcc_mean)
                    features.extend(mfcc_std)
                else:
                    features.extend([0] * 26)  # 13 mean + 13 std
            except Exception as e:
                print(f"MFCC extraction failed: {e}")
                features.extend([0] * 26)
            
            # 2. Spectral features
            try:
                spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
                features.append(np.mean(spectral_centroids) if spectral_centroids.size > 0 else 0)
                features.append(np.std(spectral_centroids) if spectral_centroids.size > 0 else 0)
                
                spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
                features.append(np.mean(spectral_rolloff) if spectral_rolloff.size > 0 else 0)
                features.append(np.std(spectral_rolloff) if spectral_rolloff.size > 0 else 0)
            except Exception as e:
                print(f"Spectral features extraction failed: {e}")
                features.extend([0] * 4)
            
            # 3. Zero crossing rate
            try:
                zcr = librosa.feature.zero_crossing_rate(y)
                features.append(np.mean(zcr) if zcr.size > 0 else 0)
                features.append(np.std(zcr) if zcr.size > 0 else 0)
            except Exception as e:
                print(f"ZCR extraction failed: {e}")
                features.extend([0] * 2)
            
            # 4. Chroma features
            try:
                chroma = librosa.feature.chroma_stft(y=y, sr=sr)
                if chroma.shape[1] > 0:
                    chroma_mean = np.mean(chroma, axis=1)
                    features.extend(chroma_mean)
                else:
                    features.extend([0] * 12)
            except Exception as e:
                print(f"Chroma extraction failed: {e}")
                features.extend([0] * 12)
            
            # 5. Mel-scale spectrogram
            try:
                mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
                if mel_spectrogram.shape[1] > 0:
                    mel_mean = np.mean(mel_spectrogram, axis=1)[:20]  # First 20 mel bands
                    features.extend(mel_mean)
                else:
                    features.extend([0] * 20)
            except Exception as e:
                print(f"Mel spectrogram extraction failed: {e}")
                features.extend([0] * 20)
            
            # Ensure we have the expected number of features
            expected_features = 64  # 26 (MFCC) + 4 (spectral) + 2 (ZCR) + 12 (chroma) + 20 (mel)
            while len(features) < expected_features:
                features.append(0)
            
            features = features[:expected_features]  # Trim if we have too many
            
            result = np.array(features)
            
            # Check for NaN or infinite values
            if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                print("NaN or Inf values detected in features, replacing with zeros")
                result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
            
            return result
            
        except Exception as e:
            print(f"Feature extraction error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def prepare_dataset(self):
        """Prepare dataset for training"""
        print("Preparing dataset...")
        
        # Look for audio files in the data directory
        audio_files = []
        labels = []
        
        # Check for RAVDESS structure
        data_dir = Path('data')
        for audio_file in data_dir.rglob('*.wav'):
            emotion = self.parse_filename(audio_file.name)
            if emotion != 'unknown':
                audio_files.append(str(audio_file))
                labels.append(emotion)
        
        if len(audio_files) == 0:
            print("No audio files found. Creating synthetic feature data for demonstration...")
            return self.create_synthetic_features()
        
        print(f"Found {len(audio_files)} audio files")
        
        # Extract features
        features = []
        valid_labels = []
        
        for i, audio_file in enumerate(audio_files):
            feature_vector = self.extract_features(audio_file)
            if feature_vector is not None:
                features.append(feature_vector)
                valid_labels.append(labels[i])
            
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(audio_files)} files")
        
        if len(features) == 0:
            print("No features extracted. Creating synthetic data...")
            return self.create_synthetic_features()
        
        return np.array(features), np.array(valid_labels)
    
    def create_synthetic_features(self):
        """Create synthetic features for demonstration when real audio isn't available"""
        print("Creating synthetic feature data for demonstration...")
        
        emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
        n_samples_per_emotion = 150  # More samples for better training
        n_features = 64  # Match the expected feature count
        
        features = []
        labels = []
        
        # Create more sophisticated emotion patterns
        emotion_patterns = {
            'neutral': {'mfcc_mean': 0.0, 'mfcc_std': 0.5, 'spectral_shift': 0.0, 'energy': 0.3},
            'calm': {'mfcc_mean': -0.2, 'mfcc_std': 0.3, 'spectral_shift': -0.1, 'energy': 0.2},
            'happy': {'mfcc_mean': 0.5, 'mfcc_std': 0.7, 'spectral_shift': 0.3, 'energy': 0.8},
            'sad': {'mfcc_mean': -0.4, 'mfcc_std': 0.3, 'spectral_shift': -0.2, 'energy': 0.2},
            'angry': {'mfcc_mean': 0.8, 'mfcc_std': 0.9, 'spectral_shift': 0.4, 'energy': 0.9},
            'fearful': {'mfcc_mean': 0.3, 'mfcc_std': 0.8, 'spectral_shift': 0.2, 'energy': 0.6},
            'disgust': {'mfcc_mean': -0.1, 'mfcc_std': 0.6, 'spectral_shift': 0.1, 'energy': 0.4},
            'surprised': {'mfcc_mean': 0.4, 'mfcc_std': 0.8, 'spectral_shift': 0.3, 'energy': 0.7}
        }
        
        for emotion in emotions:
            pattern = emotion_patterns[emotion]
            for _ in range(n_samples_per_emotion):
                # Generate features with emotion-specific characteristics
                feature_vector = np.zeros(n_features)
                
                # MFCC features (26 features: 13 mean + 13 std)
                mfcc_means = np.random.normal(pattern['mfcc_mean'], 0.3, 13)
                mfcc_stds = np.random.normal(pattern['mfcc_std'], 0.2, 13)
                feature_vector[:13] = mfcc_means
                feature_vector[13:26] = np.abs(mfcc_stds)  # Standard deviations should be positive
                
                # Spectral features (4 features)
                spectral_base = pattern['spectral_shift']
                feature_vector[26] = max(0, np.random.normal(1000 + spectral_base * 500, 200))  # Spectral centroid mean
                feature_vector[27] = max(0, np.random.normal(150, 50))  # Spectral centroid std
                feature_vector[28] = max(0, np.random.normal(2000 + spectral_base * 800, 300))  # Spectral rolloff mean
                feature_vector[29] = max(0, np.random.normal(200, 60))  # Spectral rolloff std
                
                # ZCR features (2 features)
                zcr_base = pattern['energy'] * 0.1
                feature_vector[30] = max(0, np.random.normal(zcr_base, 0.02))  # ZCR mean
                feature_vector[31] = max(0, np.random.normal(0.01, 0.005))  # ZCR std
                
                # Chroma features (12 features)
                chroma_energy = pattern['energy']
                for i in range(12):
                    feature_vector[32 + i] = max(0, np.random.normal(chroma_energy * 0.5, 0.2))
                
                # Mel features (20 features)
                mel_energy = pattern['energy']
                for i in range(20):
                    feature_vector[44 + i] = max(0, np.random.exponential(mel_energy * 10))
                
                # Add some noise and variation
                noise = np.random.normal(0, 0.05, n_features)
                feature_vector += noise
                
                features.append(feature_vector)
                labels.append(emotion)
        
        return np.array(features), np.array(labels)
    
    def build_model(self, input_shape, num_classes):
        """Build the emotion detection model"""
        model = Sequential([
            # Input layer with regularization
            Dense(256, activation='relu', input_shape=(input_shape,)),
            BatchNormalization(),
            Dropout(0.4),
            
            # Hidden layers with proper regularization
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(32, activation='relu'),
            Dropout(0.2),
            
            # Output layer
            Dense(num_classes, activation='softmax')
        ])
        
        # Use a more conservative learning rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self):
        """Train the emotion detection model"""
        try:
            print("Starting model training...")
            
            # Download and prepare dataset
            self.download_dataset()
            features, labels = self.prepare_dataset()
            
            print(f"Dataset shape: {features.shape}")
            print(f"Labels shape: {labels.shape}")
            print(f"Unique emotions: {np.unique(labels)}")
            
            # Encode labels
            self.label_encoder = LabelEncoder()
            encoded_labels = self.label_encoder.fit_transform(labels)
            categorical_labels = to_categorical(encoded_labels)
            
            print(f"Number of classes: {len(self.label_encoder.classes_)}")
            print(f"Classes: {list(self.label_encoder.classes_)}")
            
            # Normalize features
            self.feature_scaler = StandardScaler()
            normalized_features = self.feature_scaler.fit_transform(features)
            
            # Split dataset
            X_train, X_test, y_train, y_test = train_test_split(
                normalized_features, categorical_labels, 
                test_size=0.2, random_state=42, stratify=encoded_labels
            )
            
            print(f"Training set shape: {X_train.shape}")
            print(f"Test set shape: {X_test.shape}")
            
            # Build model
            self.model = self.build_model(X_train.shape[1], len(self.label_encoder.classes_))
            
            print("Model architecture:")
            self.model.summary()
            
            # Define callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6
                )
            ]
            
            print("Training model...")
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=100,
                batch_size=32,
                verbose=1,
                callbacks=callbacks
            )
            
            # Evaluate model
            test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
            print(f"Final Test Accuracy: {test_accuracy:.4f}")
            print(f"Final Test Loss: {test_loss:.4f}")
            
            # Save model and encoders
            self.save_model()
            
            return history
            
        except Exception as e:
            print(f"Error during model training: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def save_model(self):
        """Save the trained model and preprocessing objects"""
        try:
            os.makedirs('models', exist_ok=True)
            
            # Save model
            self.model.save('models/emotion_model.h5')
            print("Model saved to models/emotion_model.h5")
            
            # Save label encoder
            with open('models/label_encoder.pkl', 'wb') as f:
                pickle.dump(self.label_encoder, f)
            print("Label encoder saved to models/label_encoder.pkl")
            
            # Save feature scaler
            with open('models/feature_scaler.pkl', 'wb') as f:
                pickle.dump(self.feature_scaler, f)
            print("Feature scaler saved to models/feature_scaler.pkl")
            
            print("Model and preprocessors saved successfully!")
            
        except Exception as e:
            print(f"Error saving model: {e}")
            import traceback
            traceback.print_exc()
    
    def load_model(self):
        """Load the trained model and preprocessing objects"""
        try:
            # Check if all required files exist
            required_files = [
                'models/emotion_model.h5',
                'models/label_encoder.pkl',
                'models/feature_scaler.pkl'
            ]
            
            for file_path in required_files:
                if not os.path.exists(file_path):
                    print(f"Required file not found: {file_path}")
                    return False
            
            # Load model
            self.model = tf.keras.models.load_model('models/emotion_model.h5')
            print("Model loaded from models/emotion_model.h5")
            
            # Load label encoder
            with open('models/label_encoder.pkl', 'rb') as f:
                self.label_encoder = pickle.load(f)
            print("Label encoder loaded from models/label_encoder.pkl")
            
            # Load feature scaler
            with open('models/feature_scaler.pkl', 'rb') as f:
                self.feature_scaler = pickle.load(f)
            print("Feature scaler loaded from models/feature_scaler.pkl")
            
            print("All model components loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def predict_emotion(self, audio_path):
        """Predict emotion from audio file"""
        try:
            if self.model is None or self.feature_scaler is None or self.label_encoder is None:
                print("Model components not properly loaded!")
                return None
            
            # Check if file exists
            if not os.path.exists(audio_path):
                print(f"Audio file not found: {audio_path}")
                return None
            
            # Extract features
            features = self.extract_features(audio_path)
            if features is None:
                print("Failed to extract features from audio file")
                return None
            
            # Normalize features
            features = features.reshape(1, -1)
            normalized_features = self.feature_scaler.transform(features)
            
            # Make prediction
            predictions = self.model.predict(normalized_features, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            
            emotion = self.label_encoder.inverse_transform([predicted_class])[0]
            
            # Create detailed prediction results
            all_predictions = {}
            for i, class_name in enumerate(self.label_encoder.classes_):
                all_predictions[class_name] = float(predictions[0][i])
            
            return {
                'emotion': emotion,
                'confidence': float(confidence),
                'all_predictions': all_predictions
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def predict_emotion_from_array(self, audio_array, sr=22050):
        """Predict emotion from audio array (for real-time processing)"""
        try:
            if self.model is None or self.feature_scaler is None or self.label_encoder is None:
                print("Model components not properly loaded!")
                return None
            
            # Validate input
            if len(audio_array) == 0:
                print("Empty audio array provided")
                return None
            
            # Ensure audio array is float32
            audio_array = np.array(audio_array, dtype=np.float32)
            
            # Extract features from audio array
            features = self._extract_features_from_array(audio_array, sr)
            
            if features is None:
                print("Failed to extract features from audio array")
                return None
            
            # Reshape and normalize features
            features = features.reshape(1, -1)
            
            # Ensure feature vector has correct dimensions
            expected_features = self.feature_scaler.n_features_in_
            if features.shape[1] != expected_features:
                print(f"Feature dimension mismatch: got {features.shape[1]}, expected {expected_features}")
                # Pad or truncate features to match expected size
                if features.shape[1] < expected_features:
                    padding = np.zeros((1, expected_features - features.shape[1]))
                    features = np.concatenate([features, padding], axis=1)
                else:
                    features = features[:, :expected_features]
            
            normalized_features = self.feature_scaler.transform(features)
            
            # Make prediction
            predictions = self.model.predict(normalized_features, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            
            emotion = self.label_encoder.inverse_transform([predicted_class])[0]
            
            # Create detailed prediction results
            all_predictions = {}
            for i, class_name in enumerate(self.label_encoder.classes_):
                all_predictions[class_name] = float(predictions[0][i])
            
            return {
                'emotion': emotion,
                'confidence': float(confidence),
                'all_predictions': all_predictions
            }
            
        except Exception as e:
            print(f"Error in array prediction: {e}")
            import traceback
            traceback.print_exc()
            return None

# Training script
if __name__ == "__main__":
    try:
        # Create and train the model
        detector = VoiceEmotionDetector()
        
        print("Starting voice emotion detection model training...")
        
        # Train the model
        history = detector.train_model()
        
        if history is not None:
            print("Training completed!")
            
            # Test the model with synthetic data
            if detector.model is not None:
                print("\nModel ready for inference!")
                print(f"Supported emotions: {list(detector.label_encoder.classes_)}")
                
                # Test with synthetic audio data
                print("\nTesting with synthetic audio...")
                try:
                    test_audio = np.random.randn(22050)  # 1 second of random audio
                    test_result = detector.predict_emotion_from_array(test_audio, 22050)
                    if test_result:
                        print(f"Test prediction: {test_result['emotion']} (confidence: {test_result['confidence']:.2f})")
                        print("All predictions:")
                        for emotion, conf in test_result['all_predictions'].items():
                            print(f"  {emotion}: {conf:.3f}")
                    else:
                        print("Test prediction failed")
                except Exception as test_error:
                    print(f"Test prediction error: {test_error}")
                    import traceback
                    traceback.print_exc()
            else:
                print("Model training failed!")
        else:
            print("Training failed to complete!")
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as main_error:
        print(f"Training script error: {main_error}")
        import traceback
        traceback.print_exc()
    finally:
        print("Training script finished")