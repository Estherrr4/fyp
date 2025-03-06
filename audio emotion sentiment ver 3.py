import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os
import traceback


#add detailed error logging
def log_error(e):
    st.error("An error occurred:")
    st.error(str(e))
    st.error("Detailed error trace:")
    st.error(traceback.format_exc())

#check if files exist
def check_files():
    required_files = ['audio_sentiment_model.keras', 'encoder_label.pkl']
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    return missing_files

st.set_page_config(
    page_title="Audio Sentiment Analysis",
    page_icon="🎙️",
    layout="centered"
)

st.markdown("""
<style>
.main-title{
    font-size: 36px;
    color: #FF6B6B;
    text-align: center;
    margin-bottom: 20px;
}
.subtitle{
    font-size: 16px;
    color: #4A4A4A;
    text-align: center;
    margin-bottom: 30px
}
.stFileUploader{
    background-color: #F8F9FA;
    border: 2px dashed #E9ECEF;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
}
.predict-button{
    width: 100%;
    backgroun-color: #FF6B6B !important;
    color: white !important;
    border: none;
    border-radius: 10px;
    padding: 10px;
    font-size: 16px;
}
</style>
""", unsafe_allow_html = True)

def add_noise(data):
    noise_value = 0.015 * np.random.uniform() * np.amax(data)
    data = data + noise_value * np.random.normal(size = data.shape[0])
    return data

def stretch_process(data, rate = 0.8):
    return librosa.effects.time_stretch(y = data, rate = rate)

def pitch_process(data, sampling_rate, pitch_factor = 0.7):
    return librosa.effects.pitch_shift(y = data, sr = sampling_rate, n_steps = pitch_factor)

def extract_process(data, sample_rate):
    output_result = np.array([])

    mean_zero = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    output_result = np.hstack((output_result, mean_zero))

    stft_out = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S = stft_out, sr = sample_rate).T, axis = 0)
    output_result = np.hstack((output_result, chroma_stft))

    mfcc_out = np.mean(librosa.feature.mfcc(y = data, sr = sample_rate).T, axis = 0)
    output_result = np.hstack((output_result, mfcc_out))

    root_mean_out = np.mean(librosa.feature.rms(y = data).T, axis = 0)
    output_result = np.hstack((output_result, root_mean_out))

    mel_spectrogram = np.mean(librosa.feature.melspectrogram(y = data, sr = sample_rate).T, axis = 0)
    output_result = np.hstack([output_result, mel_spectrogram])

    return output_result

def export_process(audio_data, sample_rate):
    output_1 = extract_process(audio_data, sample_rate)
    result = np.array(output_1)

    noise_out = add_noise(audio_data)
    output_2 = extract_process(noise_out, sample_rate)
    result = np.vstack((result, output_2))

    new_out = stretch_process(audio_data)
    stretch_pitch = pitch_process(new_out, sample_rate)
    output_3 = extract_process(stretch_pitch, sample_rate)
    result = np.vstack((result, output_3))

    return result

def reshape_to_fixed_size(data, target_features = 704):
    if data.shape[1] > target_features:
        return data[:, :target_features]
    elif data.shape[1] < target_features:
        pad_width = ((0, 0), (0, target_features - data.shape[1]))
        return np.pad(data, pad_width, mode = 'constant')
    return data

def predict_emotion(audio_path):
    project_path = "C:/Users/esther/OneDrive - Tunku Abdul Rahman University College/FYP/Project II"
    os.chdir(project_path)

    loaded_model = tf.keras.models.load_model(os.path.join(project_path, 'audio_sentiment_model.keras'))
    with open(os.path.join(project_path, 'encoder_label.pkl'), 'rb') as f:
        loaded_encoder = pickle.load(f)

    # Print out preprocessing details for debugging
    st.write(f"Audio file: {audio_path}")

    data, sample_rate = librosa.load(audio_path, duration = 2.5, offset = 0.6)

    # Show audio characteristics
    st.write(f"Sample Rate: {sample_rate}")
    st.write(f"Audio Length: {len(data)}")

    features = export_process(data, sample_rate)

    # Print feature shape to verify
    #st.write(f"Feature Shape before prediction: {features.shape}")
    #if len(features.shape) == 2:
     #   features = np.expand_dims(features, axis = 0)
    #st.write(f"Feature Shape after reshaping: {features.shape}")
    scaler_data = StandardScaler()
    features = scaler_data.fit_transform(features)
    features = reshape_to_fixed_size(features)
    st.write(f"Feature Shape before prediction: {features.shape}")
    features = features.reshape(features.shape[0], features.shape[1],1)
    st.write(f"Feature Shape after reshaping: {features.shape}")
    #processed_features = processed_features.reshape(processed_features.shape[0], 704, 1)

    #predictions = loaded_model.predict(processed_features)
    try:
        predictions = loaded_model.predict(features)

    #most_confident_pred = np.argmax(predictions, axis = 1)
    #emotion = loaded_encoder.inverse_transform(most_confident_pred)[0]

    # Convert prediction to label
        predicted_indices = np.argmax(predictions, axis = 1)
        predicted_emotion = loaded_encoder.inverse_transform(predicted_indices)
        confidence_scores = np.max(predictions, axis = 1)
        emotion = predicted_emotion[0]
        emotion_classes = loaded_encoder.classes_
        all_confidence_scores = predictions[0]
        return emotion, all_confidence_scores, emotion_classes
    #confidence_scores = predictions[0]
    #emotion_classes = loaded_encoder.classes_

    #return emotion, confidence_scores, emotion_classes
        #return predicted_emotion
    except Exception as e:
        log_error(e)
        st.error(f"Prediction error: {str(e)}")
        # Additional debugging information
        st.write("Model input shape:", loaded_model.input_shape)
        st.write("Features shape:", features.shape)
        return None, None, None

def main():
    project_path = "C:/Users/esther/OneDrive - Tunku Abdul Rahman University College/FYP/Project II"
    os.chdir(project_path)

    st.markdown('<h1 class="main-title">Audio Sentiment Analysis 🎙️</h1>', unsafe_allow_html = True)
    st.markdown('<p class = "subtitle">Upload an audio file to detect emotion</p>', unsafe_allow_html = True)

    missing_files = check_files()
    if missing_files:
        st.error(f"Missing required files: {','.join(missing_files)}")
        st.error("Please make sure all model files are in the correct location.")
        return

    uploaded_file = st.file_uploader("Choose and audio file", type = ['wav', 'mp3'])

    if uploaded_file is not None:
        #save file temporary
        with open("temp_audio.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.audio(uploaded_file, format = 'audio/wav')

        try:
            emotion, confidence_scores, emotion_classes = predict_emotion("temp_audio.wav")
            st.subheader(f"Predicted Emotion: {emotion.upper()}")

            if emotion is not None:
                st.subheader(f"Predicted Emotion: {emotion.upper()}")

                #create confidence score visualization
                if emotion_classes is not None and confidence_scores is not None:
                    fig, ax = (plt.subplots(figsize = (10,5)))
                    ax.bar(emotion_classes, confidence_scores)
                    ax.set_title('Emotion Confidence Scores')
                    ax.set_xlabel('Emotion')
                    ax.set_ylabel('Conficence')
                    plt.xticks(rotation = 45)
                    st.pyplot(fig)
            #conf_df = pd.DataFrame({
             #   'Emotion' : emotion_classes,
              #  'Confidence': confidence_scores
            #})

            
            #ax.set_xticklabels(emotion_classes, rotation = 45)         
            
            #st.pyplot(fig)
        except Exception as e:
            log_error(e)
            st.error(f"Error processing audio: {str(e)}")

if __name__ == "__main__":
    main()