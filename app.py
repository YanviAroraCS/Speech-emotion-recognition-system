import streamlit as st
import numpy as np
import tensorflow as tf
import os
import librosa  # To extract speech features

# Define dataset and model paths
DATASET_PATH = r"C:\Users\yanvi\OneDrive\Desktop\RAVDESS dataset"
MODEL_PATH = os.path.join(DATASET_PATH, "mymodel.h5")

def main():
    st.title("🎤 Speech Emotion Recognition System")  # Added title here
    
    selected_box = st.sidebar.selectbox(
        "Choose an option...",
        ('Emotion Recognition', 'View Source Code')
    )

    if selected_box == 'Emotion Recognition':
        st.sidebar.success('Try it yourself by adding an audio file.')
        application()

    elif selected_box == 'View Source Code':
        st.code(open(__file__, 'r').read())  # Read the current script file

# Function to load the pre-trained model
@st.cache_resource(show_spinner=False)  # Use correct caching method for models
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

def application():
    models_load_state = st.text('Loading model...')
    model = load_model()
    models_load_state.text('Model Loaded Successfully!')

    file_to_be_uploaded = st.file_uploader("Choose an audio file...", type=["wav"])

    if file_to_be_uploaded:
        st.audio(file_to_be_uploaded, format='audio/wav')

        # Load audio as numpy array
        y, sr = librosa.load(file_to_be_uploaded, sr=22050)
        
        # Predict emotion
        emotion = predict(model, y, sr)
        st.success(f'Predicted Emotion: **{emotion}**')

def extract_mfcc(y, sr):
    """Extracts MFCC features from an audio signal."""
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfccs

def predict(model, y, sr):
    """Predicts the emotion from an audio signal."""
    emotions = {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad',
                5: 'angry', 6: 'fearful', 7: 'disgust', 8: 'surprised'}
    
    test_point = extract_mfcc(y, sr)
    test_point = np.reshape(test_point, (1, 40, 1))

    predictions = model.predict(test_point)
    predicted_label = np.argmax(predictions[0]) + 1  # Fix index shift

    return emotions[predicted_label]

if __name__ == "__main__":
    main()
