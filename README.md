# Emotion-Based Music Recommender 🎵😊

An AI-powered application that detects your emotions through facial expressions and hand gestures, then recommends music based on your current emotional state.

![Emotion Detection Demo](https://img.shields.io/badge/AI-Emotion%20Detection-blue) ![Music Recommendation](https://img.shields.io/badge/Music-Recommendation-green) ![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red)

## 🎯 Features

- **Real-time Emotion Detection**: Uses MediaPipe and machine learning to detect emotions from webcam feed
- **Facial Landmark Analysis**: Analyzes facial expressions and hand gestures for emotion recognition
- **Music Recommendation**: Automatically opens YouTube with music recommendations based on detected emotion
- **Web Interface**: Clean and intuitive Streamlit web application
- **Multi-language Support**: Search for music in different languages
- **Artist Preference**: Specify your favorite singer for personalized recommendations

## 🚀 How It Works

1. **Data Collection**: Capture facial landmarks and hand gestures for different emotions
2. **Model Training**: Train a neural network to classify emotions based on the captured features
3. **Real-time Inference**: Use the trained model to detect emotions from live webcam feed
4. **Music Recommendation**: Generate YouTube search queries based on detected emotion, language, and artist preferences

## 📁 Project Structure

```
emotion-based-music-recommender/
├── src/
│   ├── app.py              # Main Streamlit application
│   ├── data_collection.py  # Script to collect training data
│   ├── data_training.py    # Script to train the emotion classification model
│   └── inference.py        # Standalone emotion detection script
├── models/
│   ├── model.h5           # Trained Keras model
│   └── labels.npy         # Emotion labels
├── data/
│   ├── angry.npy          # Training data for angry emotion
│   ├── happy.npy          # Training data for happy emotion
│   ├── neutral.npy        # Training data for neutral emotion
│   ├── sad.npy            # Training data for sad emotion
│   └── labels.npy         # Labels for training data
├── assets/
│   └── (image files)
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── LICENSE               # MIT License
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.7 or higher
- Webcam access
- Internet connection for music recommendations

### Clone the Repository
```bash
git clone https://github.com/Kishore276/Emotion-based-music-recommender.git
cd emotion-based-music
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## 🎮 How to Run

### 1. Run the Main Application
```bash
cd src
streamlit run app.py
```

### 2. Using the Application
1. Open your web browser and navigate to the displayed URL (usually `http://localhost:8501`)
2. Enter your preferred language and singer name
3. Allow webcam access when prompted
4. Look at the camera and let the app capture your emotion
5. Click "Recommend me songs" to get music recommendations

### 3. Training Your Own Model (Optional)

#### Collect Training Data
```bash
cd src
python data_collection.py
```
- Enter emotion name when prompted (e.g., "happy", "sad", "angry", "neutral")
- Perform the emotion in front of the camera
- The script will collect 100 samples automatically

#### Train the Model
```bash
cd src
python data_training.py
```
- Make sure you have collected data for all emotions
- The script will train a neural network and save the model

#### Test Emotion Detection
```bash
cd src
python inference.py
```

## 🎭 Supported Emotions

- **Happy** 😊
- **Sad** 😢
- **Angry** 😠
- **Neutral** 😐

## 🔧 Technical Details

### Technologies Used
- **MediaPipe**: For facial landmark detection and hand tracking
- **TensorFlow/Keras**: For neural network model training and inference
- **OpenCV**: For computer vision and image processing
- **Streamlit**: For web application interface
- **streamlit-webrtc**: For real-time webcam streaming in browser

### Model Architecture
- Input layer: 1680 features (facial landmarks + hand keypoints)
- Hidden layers: 512 → 256 neurons with ReLU activation
- Output layer: Softmax activation for multi-class emotion classification

### Feature Engineering
- Facial landmarks: Relative positions normalized to nose tip
- Hand keypoints: Relative positions normalized to middle finger tip
- Missing hand data: Filled with zeros when hands are not detected

## 📈 Performance

The model achieves good accuracy on emotion classification with proper training data. Performance can be improved by:
- Collecting more diverse training data
- Adding more emotion categories
- Fine-tuning model architecture
- Using data augmentation techniques

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original concept inspired by emotion detection research
- MediaPipe team for the excellent computer vision library
- Streamlit team for making web app development simple
- YouTube for music recommendation integration

## 📞 Support

If you encounter any issues or have questions, please:
1. Check the existing issues in the repository
2. Create a new issue with detailed description
3. Include your system information and error messages

## 🔗 Links

- **Repository**: https://github.com/Kishore276/Emotion-based-music-recommender.git
<!-- - **Demo Video**: [Watch on YouTube](https://youtu.be/uDzLxos0lNU) -->
- **Issues**: Report bugs and feature requests

---

**Made with ❤️ using AI and Machine Learning**