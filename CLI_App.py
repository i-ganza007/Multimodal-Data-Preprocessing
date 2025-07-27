#!/usr/bin/env python3
"""
File-Based Multimodal CLI System
Face Recognition → Product Recommendation → Voice Authentication
Uses provided image and audio file paths
"""

import pickle
import joblib
import pandas as pd
import json
import numpy as np
from PIL import Image
import librosa
import argparse
import os
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

# Global variables to store models
face_model = None
recommendation_model = None
voice_model = None
voice_scaler = None
voice_features = None
mobilenet = None

def load_models():
    """Load all ML models using joblib"""
    global face_model, recommendation_model, voice_model, voice_scaler, voice_features, mobilenet
    
    print("Loading models...")
    
    try:
        # Load MobileNetV2 for feature extraction
        print("Loading MobileNetV2...")
        mobilenet = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
        print("✅ MobileNetV2 loaded")
        
        # Load face recognition model
        print("Loading face recognition model...")
        face_model = joblib.load('models/face_recognition_model.pkl')
        print("✅ Face model loaded")
        
        # Load recommendation model
        print("Loading recommendation model...")
        recommendation_model = joblib.load('models/xgb_model.joblib')
        print("✅ Recommendation model loaded")
        
        # Load voice models
        print("Loading voice verification model...")
        voice_model = joblib.load('models/voiceprint_verification_model.pkl')
        print("✅ Voice model loaded")
        
        print("Loading voice scaler...")
        voice_scaler = joblib.load('models/voiceprint_scaler.pkl')
        print("✅ Voice scaler loaded")
        
        print("Loading voice features...")
        voice_features = joblib.load('models/voiceprint_feature_columns.pkl')
        print("✅ Voice features loaded")
        
        print("✅ All models loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        raise e

def load_and_preprocess_image(img_path):
    """Load and preprocess image exactly like in your notebook"""
    try:
        img = Image.open(img_path).resize((224, 224))
        img_array = np.array(img)
        
        # Handle different image formats
        if img_array.ndim == 2:
            img_array = np.stack((img_array,) * 3, axis=-1)
        elif img_array.shape[-1] == 1:
            img_array = np.repeat(img_array, 3, axis=-1)
        
        img_array = preprocess_input(img_array.astype(np.float32))
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return None

def load_image(image_path):
    """Load image from file path"""
    print(f"📸 Loading image from: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        return None
    
    img_tensor = load_and_preprocess_image(image_path)
    if img_tensor is None:
        return None
    
    print("✅ Image loaded successfully!")
    return img_tensor

def recognize_face(img_tensor, threshold=0.55):
    """Process face image and return user ID using MobileNetV2 features"""
    print("🔍 Processing facial recognition...")
    
    # Extract features from MobileNet (exactly like your notebook)
    feature_vector = mobilenet.predict(img_tensor)[0].reshape(1, -1)
    print(f"Feature shape: {feature_vector.shape} (should be 1280)")
    
    # Get predictions
    probs = face_model.predict_proba(feature_vector)[0]
    predicted_index = np.argmax(probs)
    predicted_class = face_model.classes_[predicted_index]
    confidence = probs[predicted_index]
    
    print(f"🧠 Prediction Probabilities:")
    for cls, prob in zip(face_model.classes_, probs):
        print(f"  ➤ {cls}: {prob:.2f}")
    
    print(f"🎯 Predicted: {predicted_class}")
    print(f"🔐 Confidence: {confidence:.2f}")
    
    if confidence >= threshold:
        print(f"✅ Access Granted to: {predicted_class}")
        return predicted_class
    else:
        print(f"❌ Access Denied: Unknown user (confidence too low)")
        return None

def generate_recommendations(user_id):
    """Generate product recommendations for user"""
    print("🛍️ Generating product recommendations...")
    
    # Create user features for XGBoost model (9 features expected)
    # You'll need to replace this with actual user feature extraction based on your training data
    # Common features might be: user_id, age, gender, income, purchase_history, category_preferences, etc.
    
    # Example 9 features (adjust based on your actual model training)
    user_features = np.array([[
        hash(user_id) % 1000,  # user_id encoded
        25,                    # age
        1,                     # gender (0/1)
        50000,                # income
        3,                    # purchase_count
        2,                    # preferred_category
        1,                    # is_premium_user
        0.75,                 # avg_rating
        100                   # total_spent
    ]])
    
    print(f"User features shape: {user_features.shape} (should be (1, 9))")
    
    # Get recommendations
    recommendations = recommendation_model.predict(user_features)
    
    print("📋 Recommendations generated (waiting for voice verification...)")
    return recommendations

def load_audio(audio_path):
    """Load audio from file path"""
    print(f"🎤 Loading audio from: {audio_path}")
    
    if not os.path.exists(audio_path):
        print(f"❌ Audio file not found: {audio_path}")
        return None, None
    
    try:
        audio, sample_rate = librosa.load(audio_path, sr=22050)
        print("✅ Audio loaded successfully!")
        return audio, sample_rate
    except Exception as e:
        print(f"❌ Could not load audio: {e}")
        return None, None

def extract_voice_features_dict(audio_file, speaker=None):
    """
    Extract voice features as dictionary (same as training)
    """
    try:
        y, sr = librosa.load(audio_file)
        duration = librosa.get_duration(y=y, sr=sr)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta_mean = np.mean(mfcc_delta, axis=1)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        mfcc_delta2_mean = np.mean(mfcc_delta2, axis=1)

        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        rolloff_mean = np.mean(rolloff)
        rolloff_std = np.std(rolloff)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        centroid_mean = np.mean(centroid)
        centroid_std = np.std(centroid)
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        bandwidth_mean = np.mean(bandwidth)
        bandwidth_std = np.std(bandwidth)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = np.mean(contrast)
        contrast_std = np.std(contrast)
        flatness = librosa.feature.spectral_flatness(y=y)
        flatness_mean = np.mean(flatness)
        flatness_std = np.std(flatness)

        rms = librosa.feature.rms(y=y)
        rms_mean = np.mean(rms)
        rms_std = np.std(rms)
        zcr = librosa.feature.zero_crossing_rate(y=y)
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)
        f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        f0_mean = np.mean(f0[~np.isnan(f0)]) if np.any(~np.isnan(f0)) else 0
        f0_std = np.std(f0[~np.isnan(f0)]) if np.any(~np.isnan(f0)) else 0
        f0_min = np.min(f0[~np.isnan(f0)]) if np.any(~np.isnan(f0)) else 0
        f0_max = np.max(f0[~np.isnan(f0)]) if np.any(~np.isnan(f0)) else 0

        order = 12
        autocorr = np.correlate(y, y, mode='full')
        autocorr = autocorr[len(autocorr)//2:len(autocorr)//2+order+1]
        r = autocorr[1:]
        R = autocorr[:-1]
        from scipy.linalg import solve_toeplitz
        lpc_coeffs = solve_toeplitz((R, R), r)[:order]
        lpc_coeffs = np.concatenate(([1], -lpc_coeffs))

        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)

        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_mean = np.mean(onset_env)
        onset_std = np.std(onset_env)

        features = {
            'file': audio_file,
            'speaker': speaker,
            'duration': duration,
            **{f'mfcc_{i}_mean': mfcc_mean[i] for i in range(13)},
            **{f'mfcc_{i}_std': mfcc_std[i] for i in range(13)},
            **{f'mfcc_delta_{i}_mean': mfcc_delta_mean[i] for i in range(13)},
            **{f'mfcc_delta2_{i}_mean': mfcc_delta2_mean[i] for i in range(13)},
            'rolloff_mean': rolloff_mean,
            'rolloff_std': rolloff_std,
            'centroid_mean': centroid_mean,
            'centroid_std': centroid_std,
            'bandwidth_mean': bandwidth_mean,
            'bandwidth_std': bandwidth_std,
            'contrast_mean': contrast_mean,
            'contrast_std': contrast_std,
            'flatness_mean': flatness_mean,
            'flatness_std': flatness_std,
            'rms_mean': rms_mean,
            'rms_std': rms_std,
            'zcr_mean': zcr_mean,
            'zcr_std': zcr_std,
            'f0_mean': f0_mean,
            'f0_std': f0_std,
            'f0_min': f0_min,
            'f0_max': f0_max,
            **{f'lpc_{i}': lpc_coeffs[i] for i in range(13)},
            **{f'chroma_{i}_mean': chroma_mean[i] for i in range(12)},
            'onset_mean': onset_mean,
            'onset_std': onset_std
        }
        
        return features
        
    except Exception as e:
        print(f"Error extracting features: {str(e)}")
        return None

def verify_voice(audio_path, user_id):
    """Load audio file and verify voice authentication"""
    print("🔐 Starting voice authentication...")
    
    # Load audio
    audio, sample_rate = load_audio(audio_path)
    if audio is None:
        return False
    
    # Extract features using the same method as training
    print("🔊 Extracting voice features...")
    features_dict = extract_voice_features_dict(audio_path, speaker=user_id)
    
    if features_dict is None:
        print("❌ Feature extraction failed!")
        return False
    
    # Convert to DataFrame
    import pandas as pd
    feature_df = pd.DataFrame([features_dict])
    
    # Get feature columns (excluding metadata)
    all_feature_cols = [col for col in feature_df.columns 
                       if col not in ['file', 'speaker', 'command', 'duration']]
    
    print(f"Extracted {len(all_feature_cols)} voice features")
    
    # Get the features that your model was trained with
    # This should match the 68 features after variance threshold selection
    try:
        # Load the selected feature names if you saved them
        selected_features = joblib.load('models/voiceprint_feature_columns.pkl')
        X_features = feature_df[selected_features].values
        print(f"Using {len(selected_features)} selected features")
    except:
        # If you don't have the selected feature names, apply variance threshold
        from sklearn.feature_selection import VarianceThreshold
        selector = VarianceThreshold(threshold=0.01)
        X_all = feature_df[all_feature_cols].values
        X_features = selector.fit_transform(X_all)  # This will select ~68 features
        print(f"Applied variance threshold, got {X_features.shape[1]} features")
    
    # Scale features
    voice_features_scaled = voice_scaler.transform(X_features)
    
    # Verify voice
    verification_result = voice_model.predict(voice_features_scaled)[0]
    
    print(f"🔍 Voice verification result: {verification_result}")
    
    # OneClassSVM returns 1 for inlier (authorized), -1 for outlier (unauthorized)
    is_verified = verification_result == 1
    
    if is_verified:
        print(f"✅ Voice authentication successful for user: {user_id}")
    else:
        print(f"❌ Voice authentication failed for user: {user_id}")
    
    return is_verified

def display_recommendations(recommendations):
    """Display the final recommendations"""
    print("\n" + "="*50)
    print("🎉 VOICE VERIFICATION SUCCESSFUL!")
    print("📋 YOUR PERSONALIZED RECOMMENDATIONS:")
    print("="*50)

    try:
        categories = pd.read_csv('Datasets/merged_dataset.csv')
        label_map = dict(enumerate(categories['product_category'].unique()))
        
        for i, rec in enumerate(recommendations, 1):
            category_name = label_map.get(rec, "Unknown Category")
            print(f"{i}. Product recommendation: {category_name}")
    except Exception as e:
        print(f"❌ Could not load product categories: {e}")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. Product recommendation ID: {rec}")
    
    print("="*50)


def main():
    """Main application flow"""
    parser = argparse.ArgumentParser(description='Multimodal CLI Recommendation System')
    parser.add_argument('--image', '-i', required=True, help='Path to face image file')
    parser.add_argument('--audio', '-a', required=True, help='Path to voice audio file')
    
    args = parser.parse_args()
    
    print("🚀 Multimodal CLI Recommendation System")
    print("="*50)
    print(f"📸 Image: {args.image}")
    print(f"🎤 Audio: {args.audio}")
    print("="*50)
    
    # Load models
    try:
        load_models()
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        print("Make sure all model files are in the 'models' directory")
        return
    
    # Step 1: Face Recognition
    print("\n📍 Step 1: Facial Recognition")
    img_tensor = load_image(args.image)
    if img_tensor is None:
        return
    
    user_id = recognize_face(img_tensor)
    if user_id is None:
        print("❌ Face recognition failed or confidence too low")
        return
    
    # Step 2: Generate Recommendations
    print("\n📍 Step 2: Generating Recommendations")
    recommendations = generate_recommendations(user_id)
    
    # Step 3: Voice Authentication
    print("\n📍 Step 3: Voice Authentication")
    is_verified = verify_voice(args.audio, user_id)
    
    if is_verified:
        display_recommendations(recommendations)
    else:
        print("❌ Voice verification failed! Access denied.")
        print("🔒 Recommendations are locked.")

if __name__ == "__main__":
    main()