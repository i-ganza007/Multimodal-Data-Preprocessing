# Multimodal-Data-Preprocessing

## Overview

This project provides a comprehensive multimodal data preprocessing and machine learning pipeline for face recognition, product recommendation, and voice authentication. It integrates image, audio, and tabular data to deliver a secure, personalized recommendation system with advanced biometric verification. The project includes data processing scripts, machine learning models, and a command-line interface (CLI) for end-to-end workflow execution.

## Team Members

**Eddy Gasana**  

**Ian Ganza**  

**Placide Kabisa Manzi**  

**Lievin Murayire**  

**Rene Pierre Ntabahana**  


## Project Structure

```
Multimodal-Data-Preprocessing/
├── 📁 Datasets/
│   ├── audio_features.csv          # Processed audio features
│   ├── image_features.csv          # Processed image features
│   ├── merged_dataset.csv          # Combined multimodal dataset
│   └── 📁 Sounds/                  # Sample audio files for testing
│       ├── confirm_eddy.wav
│       ├── confirm_ian.wav
│       ├── confirm_lievin.wav
│       ├── confirm_placide.wav
│       ├── isaac.wav
│       ├── yes_approve_eddy.wav
│       ├── yes_approve_ian.wav
│       ├── yes_approve_lievin.wav
│       └── yes_approve_placide.wav
├── 📁 models/
│   ├── face_recognition_model.pkl       # Trained face recognition classifier
│   ├── voiceprint_feature_columns.pkl  # Selected voice features
│   ├── voiceprint_model_metadata.json  # Voice model metadata
│   ├── voiceprint_model.pkl            # Voiceprint verification model
│   ├── voiceprint_scaler.pkl           # Feature scaler for voice data
│   ├── voiceprint_verification_model.pkl # OneClassSVM for voice verification
│   └── xgb_model.joblib                # XGBoost recommendation model
├── 📁 notebooks/
│   ├── Data_merge.ipynb                 # Data merging and exploration
│   ├── Facial_Recognition_Model.ipynb  # Face recognition model training
│   ├── Image_Processing_.ipynb         # Image preprocessing workflows
│   ├── Product_Recommendation_Model.ipynb # Recommendation system training
│   ├── Sound_Processing.ipynb          # Audio feature extraction
│   └── VoicePrint_Model.ipynb         # Voice authentication model
├── 📁 scripts/
│   └── CLI_App.py                      # Main CLI application
├── 📁 utils/
│   └── known_features.pkl              # Known face embeddings
├── requirements.txt                     # Project dependencies
├── LICENSE                             # Project license
└── README.md                           # Project documentation
```

## Key Features

###  **Multimodal Authentication Pipeline**
- **Facial Recognition**: Utilizes MobileNetV2 for feature extraction with custom classifier
- **Voice Authentication**: Advanced voiceprint verification using OneClassSVM
- **Product Recommendation**: XGBoost-based personalized recommendation engine

###  **Security Features**
- Dual-factor biometric authentication (face + voice)
- Distance-based verification with configurable thresholds
- Secure feature extraction and model inference

###  **Data Processing Capabilities**
- Comprehensive audio feature extraction (MFCC, spectral features, LPC coefficients)
- Image preprocessing with MobileNetV2 feature embeddings
- Advanced data merging and feature engineering

###  **Command-Line Interface**
- End-to-end pipeline execution from command line
- File-based input processing for images and audio
- Real-time authentication and recommendation delivery

## Installation & Setup

### Prerequisites
- Python 3.7 or higher
- Git

### Installation Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/i-ganza007/Multimodal-Data-Preprocessing.git
   cd Multimodal-Data-Preprocessing
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify Model Files**
   Ensure all required model files are present in the `models/` directory.

## Usage Guide

### CLI Application

Run the complete multimodal pipeline:

```bash
python scripts/CLI_App.py --image path/to/your/image.jpg --audio path/to/your/audio.wav
```

#### Pipeline Flow:
1. **Step 1**: Facial recognition and user identification
2. **Step 2**: Product recommendation generation
3. **Step 3**: Voice authentication verification
4. **Step 4**: Secure recommendation delivery (if voice verified)

### Example Usage

```bash
# Example with sample files
python scripts/CLI_App.py --image Datasets/sample_face.jpg --audio Datasets/Sounds/confirm_eddy.wav
```

## Technical Implementation

### Machine Learning Models

| Component | Algorithm | Purpose |
|-----------|-----------|---------|
| Face Recognition | MobileNetV2 + Logistic Regression | User identification from facial features |
| Voice Authentication | OneClassSVM | Voiceprint verification and security |
| Product Recommendation | XGBoost | Personalized product suggestions |

### Feature Engineering

**Audio Features (68 selected features):**
- MFCC coefficients and derivatives
- Spectral features (rolloff, centroid, bandwidth, contrast)
- Temporal features (RMS, ZCR, onset strength)
- Fundamental frequency (F0) statistics
- Linear Predictive Coding (LPC) coefficients
- Chroma features

**Image Features:**
- MobileNetV2 embeddings (1280-dimensional)
- Preprocessed and normalized feature vectors

## Jupyter Notebooks

| Notebook | Description |
|----------|-------------|
| `Data_merge.ipynb` | Data integration and exploratory analysis |
| `Facial_Recognition_Model.ipynb` | Face recognition model development |
| `Image_Processing_.ipynb` | Image preprocessing and feature extraction |
| `Product_Recommendation_Model.ipynb` | Recommendation system training |
| `Sound_Processing.ipynb` | Audio analysis and feature extraction |
| `VoicePrint_Model.ipynb` | Voice authentication model training |

## Configuration & Customization

### Model Thresholds
- **Face Recognition Confidence**: 0.55 (adjustable in CLI_App.py)
- **Face Distance Threshold**: 0.6 (adjustable in CLI_App.py)
- **Voice Verification**: OneClassSVM decision boundary

### Adding New Users
1. Collect face images and voice samples
2. Extract features using provided notebooks
3. Retrain models with new data
4. Update known_features.pkl with new embeddings

## Dependencies

Key libraries and frameworks:
- **Machine Learning**: TensorFlow, Keras, scikit-learn, XGBoost, LightGBM
- **Audio Processing**: librosa, audiomentations
- **Image Processing**: PIL, OpenCV, matplotlib
- **Data Manipulation**: pandas, numpy
- **Model Persistence**: joblib, pickle

See `requirements.txt` for complete dependency list.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Future Enhancements

- [ ] Real-time webcam and microphone integration
- [ ] REST API development for web applications
- [ ] Mobile application support
- [ ] Enhanced security with additional biometric modalities
- [ ] Cloud deployment and scalability improvements

## License

This project is licensed under the terms specified in the `LICENSE` file.

## Contact & Support

For questions, issues, or collaboration opportunities, please contact any of the team members listed above or create an issue in this repository.

---

**Note**: Replace the placeholder names and email addresses in the "Team Members" section with actual team member information.