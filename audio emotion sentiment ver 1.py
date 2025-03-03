import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import pickle
import os
from sklearn.preprocessing import LabelEncoder


project_path = "C:/Users/esther/OneDrive - Tunku Abdul Rahman University College/FYP/Project II"
os.chdir(project_path) #change working directory
#st.write("Current Working Directory: ", os.getcwd())


# Load pre-trained model (ensure the model file is in the same directory)
@st.cache_resource
def load_model():
    model_path = os.path.join(os.getcwd(), "audio_sentiment_model.keras")
    

    # Check if the model file exists before loading
    if not os.path.exists(model_path):
        st.error(f"Error: Model file '{model_path}' not found. Please ensure the file exists in the project directory.")
        return None 

    model = tf.keras.models.load_model(model_path)
    #return tf.keras.models.load_model(model_path)

    print("Model loaded successfully!")
    print("Model expected input shape:", model.input_shape)
    return model
model = load_model()

# Load label encoder with error handling
encoder_label = None
#encoder_label - LabelEncoder()
#encoder_label.fit(OneHotEncoder())
encoder_path = os.path.join(os.getcwd(), "encoder_label.pkl")

if os.path.exists(encoder_path):
    with open("encoder_label.pkl", "rb") as f:
        encoder_label = pickle.load(f)
    #st.write("Label encoder loaded successfully.")

    print("Loaded object type:", type(encoder_label))
    if hasattr(encoder_label, "classes_"):
        print("This is a valid LabelEncoder. Classes:", encoder_label.classes_)
    elif hasattr(encoder_label, "categories_"):
        print("This is a OneHotEncoder, not a LabelEncoder. Categories:", encoder_label.categories_)
    else:
        print("Unknown object type. It is NOT a LabelEncoder.")

    if isinstance(encoder_label, LabelEncoder):
        st.write("Label encoder loaded successfully.")
        st.write(f"Classes: {encoder_label.classes_}")
    else:
        st.error("Error: encoder_label.pkl is not a LabelEncoder! Retrain and save it as a LabelEncoder.")
        encoder_label = None
else:
    st.error("Error: label encoder file 'encoder_label.pkl' not found")
    



# Function to extract features 
def extract_features(audio_path):
    num_mfcc = 40
    num_chroma = 12
    num_mel = 128
    data, sample_rate = librosa.load(audio_path, duration=2.5, offset=0.6)

    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=num_mfcc).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=data, sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    
    features = np.hstack([mfcc, chroma, mel])
    print("Features shape before reshape: ", features.shape)
    
    expected_shape = 162
    #adjust feature shape if necessary
    if features.shape[0] > expected_shape:
        features = features[:expected_shape]  # Trim excess
    elif features.shape[0] < expected_shape:
        features = np.pad(features, (0, expected_shape - features.shape[0]), mode='constant')
    #print("Feature shape after adjustment: ", features.shape)
    #current_shape = features.shape[0]

    #if current_shape < expected_shape:
        #features = np.pad(features, (0, expected_shape - current_shape), mode = 'constant')
    #elif current_shape > expected_shape:
        #features = features[:expected_shape]   

    #st.write(f"Model expected input shape: {model.input_shape}")
    #st.write(f"Feature shape: {features.shape}")
    print("Final Features Shape: ", features.shape)
    #return features.reshape(1, 704)
    return features.reshape(1, 162, 1)  # Reshape for CNN input
    #return features.reshape(1, -1)

# Streamlit UI
st.title("🎵 Audio Sentiment Analysis")

# File uploader
uploaded_file = st.file_uploader("Upload an audio file (WAV, MP3)", type=["wav", "mp3"])

if uploaded_file:
    st.audio(uploaded_file, format='audio/wav')
    
    # Save uploaded file temporarily
    file_path = f"temp_audio.{uploaded_file.type.split('/')[1]}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    # Perform sentiment analysis
    def analyze_audio_sentiment(file_path):

        
        if model is None:
            st.error("Error: Model failed to load.")
            return "Unknown", 0.0
        
        if encoder_label is None:
            st.error("Error: Label encoder not loaded.")
            return "Unknown", 0.0
       
        #if features.shape != (1, 704, 1):
         #   st.error(f"Feature shape mismatch! Expected (1, 704, 1), got {features.shape}")
          #  return "Unknown", 0.0
        features = extract_features(file_path)
        print("Feature shape before prediction: ", features.shape)
        #features = features.reshape(1, -1, 1)
        #if features is None:
            #st.error("Error: Feature extraction failed.")
            #return "Unknown", 0.0
        
        
        prediction = model.predict(features)
        #st.write(f"Raw model predictions: {prediction}")

        predicted_label_index = np.argmax(prediction)

        #num_classes = 7
        #one_hot_vector = np.zeros((1, num_classes))
        #one_hot_vector[0, predicted_label_index] = 1
        #sentiment = encoder_label.inverse_transform(np.array([predicted_label_index]).reshape(1,-1))[0]
        print("Predicted raw output: ", predicted_label_index)
        
        st.write(f"Encoder type: {type(encoder_label)}")  # Print the encoder type

        #if hasattr(encoder_label, "categories_"):
         #   st.write(f"OneHotEncoder categories: {encoder_label.categories_}")
        #elif hasattr(encoder_label, "classes_"):
         #   st.write(f"LabelEncoder classes: {encoder_label.classes_}")
        #else:
         #   st.write("Unknown encoder type")

        
        #st.write(f"Classes in label encoder: {encoder_label.classes_}")
        #sentiment = encoder_label.inverse_transform([predicted_label_index])[0]
        #sentiment = encoder_label.inverse_transform(one_hot_vector)[0]

        try:
            sentiment = encoder_label.inverse_transform([predicted_label_index])[0]
        except Exception as e:
            st.error(f"Error: Failed to decode label. {e}")
            return "Unknown", 0.0


        #score = prediction[0][predicted_label_index]
        score = np.max(prediction)

        return sentiment,score
    sentiment, score = analyze_audio_sentiment(file_path)

    #st.write(f"DEBUG: Sentiment = {sentiment}, Score = {score}")

    # Display results
    st.subheader("Analysis Result:")
    st.write(f"**Sentiment:** {sentiment}")
    st.write(f"**Score:** {score:.4f}")

