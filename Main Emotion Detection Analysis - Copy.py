import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import datetime
import time
import string
import hashlib
import sqlite3
import json
import os
from pathlib import Path
import librosa
import tensorflow as tf
import pickle
from sklearn.preprocessing import StandardScaler
import traceback
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Create database directory if it doesn't exist
if not os.path.exists("data"):
    os.makedirs("data")

# Initialize database
def init_db():
    conn = sqlite3.connect('data/users.db')
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS users
    (id INTEGER PRIMARY KEY, 
    username TEXT UNIQUE, 
    password TEXT,
    history TEXT)
    ''')
    conn.commit()
    conn.close()

def hash_password(password):
    """Hash a password for storing."""
    return hashlib.sha256(str.encode(password)).hexdigest()

def verify_password(stored_password, provided_password):
    """Verify a stored password against one provided by user"""
    return stored_password == hashlib.sha256(str.encode(provided_password)).hexdigest()

def add_user(username, password):
    """Add a user to the database"""
    conn = sqlite3.connect('data/users.db')
    c = conn.cursor()
    try:
        hashed_password = hash_password(password)
        c.execute("INSERT INTO users (username, password, history) VALUES (?, ?, ?)", 
                  (username, hashed_password, json.dumps([])))
        conn.commit()
        result = True
    except sqlite3.IntegrityError:
        result = False
    finally:
        conn.close()
    return result

def authenticate_user(username, password):
    """Check if username/password combination is valid"""
    conn = sqlite3.connect('data/users.db')
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    conn.close()
    
    if result is not None:
        return verify_password(result[0], password)
    return False

def add_to_history(username, entry):
    """Add an analysis to user history"""
    conn = sqlite3.connect('data/users.db')
    c = conn.cursor()
    
    # Get current history
    c.execute("SELECT history FROM users WHERE username = ?", (username,))
    history_json = c.fetchone()[0]
    history = json.loads(history_json)
    
    # Add new entry
    history.append(entry)
    
    # Keep only last 10 entries
    if len(history) > 10:
        history = history[-10:]
    
    # Update history
    c.execute("UPDATE users SET history = ? WHERE username = ?", 
              (json.dumps(history), username))
    conn.commit()
    conn.close()

def get_history(username):
    """Get user's analysis history"""
    conn = sqlite3.connect('data/users.db')
    c = conn.cursor()
    c.execute("SELECT history FROM users WHERE username = ?", (username,))
    history_json = c.fetchone()[0]
    conn.close()
    return json.loads(history_json)

class EmotionAnalyzer:
    """Deep learning based emotion analyzer"""
    def __init__(self, model_path, tokenizer_path, encoder_path, max_len_path):
        # Load model
        try:
            self.model = tf.keras.models.load_model(model_path)
            # Load tokenizer
            with open(tokenizer_path, 'rb') as handle:
                self.tokenizer = pickle.load(handle)
            # Load label encoder
            with open(encoder_path, 'rb') as handle:
                self.label_encoder = pickle.load(handle)
            # Load max_len
            with open(max_len_path, 'r') as f:
                self.max_len = int(f.read())
                
            self.model_loaded = True
            print("ML model and components loaded successfully.")
        except Exception as e:
            self.model_loaded = False
            print(f"Error loading ML model: {str(e)}")
            # Fall back to a simple model if trained model can't be loaded
            self._init_fallback_model()
    
    def _init_fallback_model(self):
        """Initialize a fallback emotion dictionary if ML model fails to load"""
        # Define emotion lexicons as a fallback
        self.emotion_lexicon = {
            'joy': ['happy', 'joy', 'delighted', 'excited', 'pleased', 'glad', 'satisfied', 'cheerful', 'thrilled', 'ecstatic', 'elated', 'jubilant', 'gleeful', 'blissful', 'content', 'enjoy', 'smile', 'laugh', 'love', 'awesome', 'wonderful', 'fantastic', 'amazing'],
            'sadness': ['sad', 'unhappy', 'depressed', 'miserable', 'heartbroken', 'gloomy', 'melancholy', 'down', 'blue', 'dejected', 'despondent', 'distressed', 'grief', 'sorrow', 'glum', 'tearful', 'upset', 'despair', 'hurt', 'lonely', 'hopeless', 'helpless', 'disappointed'],
            'anger': ['angry', 'mad', 'furious', 'outraged', 'annoyed', 'irritated', 'enraged', 'hate', 'hatred', 'rage', 'indignant', 'exasperated', 'infuriated', 'fuming', 'irate', 'livid', 'offended', 'bitter', 'resentment', 'vexed', 'frustrated', 'cross', 'hostile'],
            'fear': ['afraid', 'scared', 'frightened', 'terrified', 'anxious', 'nervous', 'worried', 'alarmed', 'panicked', 'petrified', 'spooked', 'startled', 'apprehensive', 'cautious', 'dread', 'unease', 'concern', 'distress', 'horror', 'troubled', 'phobia', 'terror', 'panic'],
            'surprise': ['surprised', 'astonished', 'amazed', 'shocked', 'stunned', 'unexpected', 'wow', 'whoa', 'speechless', 'bewildered', 'dumbfounded', 'flabbergasted', 'staggered', 'startled', 'astounded', 'taken aback', 'awestruck', 'wonder', 'disbelief', 'incredulous'],
            'disgust': ['disgusted', 'revulsion', 'repulsed', 'aversion', 'nauseated', 'sickened', 'gross', 'yuck', 'ugh', 'nasty', 'repugnant', 'repellent', 'loathing', 'distaste', 'abhorrence', 'odious', 'offensive', 'revolting', 'distasteful', 'unpleasant', 'objectionable'],
            'shame': ['ashamed', 'guilty', 'embarrassed', 'humiliated', 'shameful', 'regretful', 'remorseful', 'mortified', 'apologetic', 'disgraced', 'dishonor', 'degraded', 'sheepish', 'self-conscious', 'sorry', 'uncomfortable', 'fool', 'blame', 'contrite', 'penitent'],
            'neutral': ['maybe', 'perhaps', 'possibly', 'probably', 'seems', 'appears', 'generally', 'occasionally', 'sometimes', 'often', 'usual', 'normal', 'typical', 'regular', 'common', 'standard', 'average', 'moderate', 'fair', 'rational', 'reasonable', 'balanced', 'ok']
        }
        
        # Add more common emotional phrases and context words
        self.contextual_phrases = {
            'joy': ['having fun', 'good time', 'great day', 'best ever', 'made my day', 'feel amazing', 'feeling blessed', 'so happy that', 'brightened my day', 'can\'t stop smiling', 'enjoy myself', 'feeling good'],
            'sadness': ['feel down', 'going through a rough time', 'lost a', 'missing you', 'broke up', 'passed away', 'not feeling well', 'hurts so much', 'feel empty', 'difficult time', 'so alone', 'crying myself'],
            'anger': ['can\'t stand', 'fed up with', 'sick and tired', 'so annoying', 'drives me crazy', 'pisses me off', 'losing my temper', 'make me mad', 'getting on my nerves', 'hate when', 'so frustrating'],
            'fear': ['worried about', 'scared that', 'afraid of', 'fear that', 'terrified of', 'panic attack', 'nervous about', 'anxiety is', 'stressing over', 'concerned about', 'dreading the'],
            'surprise': ['can\'t believe', 'never expected', 'who would have thought', 'didn\'t see that coming', 'out of nowhere', 'plot twist', 'to my surprise', 'caught me off guard'],
            'disgust': ['makes me sick', 'can\'t stomach', 'so gross', 'disgusting behavior', 'revolting that', 'absolutely repulsive', 'nauseating to', 'how disgusting'],
            'shame': ['so embarrassed', 'made a mistake', 'feel stupid', 'shouldn\'t have', 'regret doing', 'wish I hadn\'t', 'feel awful about', 'my fault', 'take the blame']
        }
        
        self.negation_words = ['not', 'no', 'never', 'neither', 'nor', "don't", "doesn't", "didn't", "won't", "wouldn't", "can't", "cannot", "couldn't", "shouldn't", "isn't", "aren't", "wasn't", "weren't"]
        self.intensifiers = ['very', 'really', 'extremely', 'absolutely', 'completely', 'totally', 'utterly', 'deeply', 'immensely', 'incredibly', 'so', 'too', 'quite', 'rather', 'somewhat']
    
    def preprocess_text(self, text):
        """Preprocess text for the ML model"""
        # Use the same preprocessing from your Jupyter notebook
        if not text:
            return ""
            
        text = str(text).lower()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Remove emojis and special characters
        text = text.encode('ascii', 'ignore').decode('ascii')
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        return text
    
    def _fallback_analyze(self, text):
        """Fall back to advanced rule-based analysis if ML model fails to load"""
        processed_text = self.preprocess_text(text)
        tokens = processed_text.split()
        
        # Count emotions
        emotion_counts = {emotion: 0 for emotion in self.emotion_lexicon.keys()}
        contributing_words = {emotion: [] for emotion in self.emotion_lexicon.keys()}
        
        # Enhanced analysis with weighted context detection
        window_size = 3  # Consider words in proximity
        
        # First pass: Check for contextual phrases
        for emotion, phrases in self.contextual_phrases.items():
            for phrase in phrases:
                if phrase in processed_text:
                    emotion_counts[emotion] += 2  # Give extra weight to phrases
                    contributing_words[emotion].append((phrase, 2))
        
        # Second pass: Process individual tokens with contextual awareness
        for i, token in enumerate(tokens):
            # Check if this is a negation
            negated = False
            intensified = False
            
            # Look back for negations and intensifiers in a window
            start_window = max(0, i - window_size)
            context_window = tokens[start_window:i]
            
            for neg_word in self.negation_words:
                if neg_word in context_window:
                    negated = True
                    break
                    
            for intensifier in self.intensifiers:
                if intensifier in context_window:
                    intensified = True
                    break
            
            # Check each emotion lexicon
            for emotion, words in self.emotion_lexicon.items():
                for word in words:
                    # Improved matching for word stems
                    if token == word or token.startswith(word):
                        weight = 1
                        
                        if intensified:
                            weight = 1.5
                            
                        if negated:
                            # If negated, reverse the emotion
                            if emotion in ['joy', 'surprise']:
                                # Negated positive emotions often indicate sadness
                                emotion_counts['sadness'] += weight
                                contributing_words['sadness'].append((f"not {token}", weight))
                            elif emotion in ['sadness', 'fear']:
                                # Negated negative emotions might indicate relief/joy
                                emotion_counts['joy'] += weight
                                contributing_words['joy'].append((f"not {token}", weight))
                            elif emotion == 'anger':
                                # Negated anger might be neutral or mild
                                emotion_counts['neutral'] += weight
                                contributing_words['neutral'].append((f"not {token}", weight))
                        else:
                            emotion_counts[emotion] += weight
                            contributing_words[emotion].append((token, weight))
        
        # If no emotions detected, use neutral
        if sum(emotion_counts.values()) == 0:
            emotion_counts['neutral'] = 1
        
        # Calculate probabilities
        total = sum(emotion_counts.values())
        emotion_probs = {emotion: count/total for emotion, count in emotion_counts.items()}
        
        # Get primary emotion
        primary_emotion = max(emotion_probs.items(), key=lambda x: x[1])[0]
        primary_score = emotion_probs[primary_emotion]
        
        # Check for mixed emotions
        sorted_emotions = sorted(emotion_probs.items(), key=lambda x: x[1], reverse=True)
        mixed_emotion = None
        if len(sorted_emotions) > 1:
            secondary_emotion = sorted_emotions[1][0]
            secondary_score = sorted_emotions[1][1]
            if secondary_score > 0.3:  # If second emotion is significant
                mixed_emotion = secondary_emotion
        
        # Word importance for visualization
        word_importance = {}
        for emotion, words in contributing_words.items():
            for word, weight in words:
                if word in word_importance:
                    word_importance[word] += weight
                else:
                    word_importance[word] = weight
        
        # Simple sentiment analysis
        positive_emotions = ['joy', 'surprise']
        negative_emotions = ['sadness', 'anger', 'fear', 'disgust', 'shame']
        
        pos_score = sum(emotion_probs[e] for e in positive_emotions)
        neg_score = sum(emotion_probs[e] for e in negative_emotions)
        neu_score = emotion_probs.get('neutral', 0)
        
        sentiment = {
            'pos': pos_score,
            'neg': neg_score,
            'neu': neu_score,
            'compound': pos_score - neg_score
        }
        
        return {
            'text': text,
            'processed_text': processed_text,
            'predicted_emotion': primary_emotion,
            'mixed_with': mixed_emotion,
            'confidence': primary_score,
            'emotion_scores': emotion_counts,
            'emotion_probabilities': emotion_probs,
            'sentiment_values': sentiment,
            'emotional_keywords': contributing_words,
            'contributing_words': contributing_words,
            'word_importance': word_importance
        }
    
    def analyze_text(self, text):
        """Analyze text using the ML model or fallback to rule-based if model not loaded"""
        if not self.model_loaded:
            # If ML model failed to load, use fallback rule-based analyzer
            st.warning("Using enhanced rule-based analysis because ML model could not be loaded.")
            return self._fallback_analyze(text)
        
        # Preprocess the text
        processed_text = self.preprocess_text(text)
        
        # Tokenize and pad
        sequence = self.tokenizer.texts_to_sequences([processed_text])
        padded_sequence = pad_sequences(sequence, maxlen=self.max_len)
        
        # Get prediction
        prediction = self.model.predict(padded_sequence)[0]
        
        # Get the predicted class index and label
        predicted_idx = np.argmax(prediction)
        predicted_emotion = self.label_encoder.inverse_transform([predicted_idx])[0]
        confidence = float(prediction[predicted_idx])
        
        # Get all emotion probabilities
        emotion_scores = {}
        for i, emotion in enumerate(self.label_encoder.classes_):
            emotion_scores[emotion] = float(prediction[i])
        
        # Check for mixed emotions
        sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
        mixed_emotion = None
        if len(sorted_emotions) > 1:
            secondary_emotion = sorted_emotions[1][0]
            secondary_score = sorted_emotions[1][1]
            if secondary_score > (confidence * 0.7):  # If second emotion is significant
                mixed_emotion = secondary_emotion
        
        # Calculate sentiment (positive/negative)
        # Adapt this list to match your model's emotion categories
        positive_emotions = ['joy', 'happy', 'surprise', 'pleasant']
        negative_emotions = ['sadness', 'anger', 'fear', 'disgust', 'shame', 'sad', 'angry']
        
        pos_score = sum(emotion_scores.get(e, 0) for e in positive_emotions)
        neg_score = sum(emotion_scores.get(e, 0) for e in negative_emotions)
        neu_score = emotion_scores.get('neutral', 0)
        
        sentiment = {
            'pos': pos_score,
            'neg': neg_score,
            'neu': neu_score,
            'compound': pos_score - neg_score
        }
        
        # Create word importance for visualization based on tokenizer
        word_importance = {}
        words = processed_text.split()
        for word in words:
            if word in self.tokenizer.word_index:
                # Basic importance based on token index (more common words have lower indices)
                max_index = len(self.tokenizer.word_index)
                normalized_importance = 1 - (self.tokenizer.word_index[word] / max_index)
                word_importance[word] = normalized_importance * 2  # Scale for better visualization
        
        # Format emotional_keywords for display
        emotional_keywords = {}
        # Group most important words by predicted emotion
        emotional_keywords[predicted_emotion] = [(word, importance) 
                                              for word, importance in sorted(word_importance.items(), 
                                                                           key=lambda x: x[1], 
                                                                           reverse=True)[:5]]
        if mixed_emotion:
            emotional_keywords[mixed_emotion] = [(word, importance) 
                                               for word, importance in sorted(word_importance.items(), 
                                                                             key=lambda x: x[1], 
                                                                             reverse=True)[5:8]]
        
        # Return the analysis results
        return {
            'text': text,
            'processed_text': processed_text,
            'predicted_emotion': predicted_emotion,
            'mixed_with': mixed_emotion,
            'confidence': confidence,
            'emotion_scores': emotion_scores,
            'emotion_probabilities': emotion_scores,
            'sentiment_values': sentiment,
            'emotional_keywords': emotional_keywords,
            'contributing_words': emotional_keywords,
            'word_importance': word_importance
        }
    
    def explain_analysis(self, analysis):
        """Create a human-readable explanation of the analysis"""
        primary_emotion = analysis['predicted_emotion']
        mixed_emotion = analysis['mixed_with']
        confidence = analysis['confidence']
        
        if mixed_emotion:
            explanation = f"Primary Emotion: {primary_emotion.upper()} mixed with {mixed_emotion.upper()}\n"
            explanation += f"Confidence: {confidence:.2f}\n\n"
        else:
            explanation = f"Primary Emotion: {primary_emotion.upper()}\n"
            explanation += f"Confidence: {confidence:.2f}\n\n"
        
        # Top three emotions
        explanation += "Top Emotions Detected:\n"
        sorted_emotions = sorted(analysis['emotion_probabilities'].items(), key=lambda x: x[1], reverse=True)[:3]
        for emotion, prob in sorted_emotions:
            explanation += f"- {emotion.capitalize()}: {prob:.2f}\n"
        
        # Sentiment analysis
        sentiment = analysis['sentiment_values']
        explanation += f"\nSentiment Analysis:\n"
        explanation += f"- Positive: {sentiment['pos']:.2f}\n"
        explanation += f"- Neutral: {sentiment['neu']:.2f}\n"
        explanation += f"- Negative: {sentiment['neg']:.2f}\n"
        
        # Emotional words
        if analysis['emotional_keywords']:
            explanation += f"\nEmotional Keywords:\n"
            for emotion, words in analysis['emotional_keywords'].items():
                if words:  # If we have words for this emotion
                    explanation += f"- {emotion.capitalize()}: "
                    emotion_words = [f"{word}" for word, _ in words[:3]]  # Take top 3
                    explanation += ", ".join(emotion_words) + "\n"
        
        # Overall explanation
        explanation += f"\nSummary:\n"
        
        # Customized explanation based on emotion
        emotion_explanations = {
            'joy': "The text expresses positive sentiments and happiness.",
            'sadness': "The text conveys feelings of sadness or melancholy.",
            'anger': "The text shows signs of frustration or anger.",
            'fear': "The text indicates anxiety or fearfulness.",
            'surprise': "The text expresses astonishment or unexpected reactions.",
            'disgust': "The text conveys feelings of aversion or repulsion.",
            'shame': "The text indicates embarrassment or shame.",
            'neutral': "The text appears to be fairly neutral in emotional content.",
            'happy': "The text expresses positive sentiments and happiness.",
            'sad': "The text conveys feelings of sadness or melancholy.",
            'angry': "The text shows signs of frustration or anger.",
            'pleasant': "The text conveys pleasant feelings and positivity."
        }
        
        # Mixed emotion explanation
        if mixed_emotion:
            if primary_emotion in ['anger', 'angry'] and mixed_emotion in ['sadness', 'sad']:
                explanation += "The text expresses anger mixed with sadness, suggesting feelings of hurt or betrayal."
            elif primary_emotion in ['sadness', 'sad'] and mixed_emotion in ['anger', 'angry']:
                explanation += "The text expresses sadness with underlying anger, suggesting disappointment or feeling let down."
            elif primary_emotion in ['fear', 'scared'] and mixed_emotion in ['sadness', 'sad']:
                explanation += "The text expresses fear combined with sadness, suggesting a sense of helplessness or despair."
            elif primary_emotion in ['joy', 'happy'] and mixed_emotion in ['surprise']:
                explanation += "The text expresses joy with a sense of surprise, suggesting unexpected happiness or pleasant astonishment."
            elif primary_emotion in ['surprise'] and mixed_emotion in ['fear', 'scared']:
                explanation += "The text expresses surprise with underlying fear, suggesting a startling or alarming revelation."
            else:
                explanation += f"The text primarily shows {primary_emotion} with elements of {mixed_emotion}."
        else:
            explanation += emotion_explanations.get(primary_emotion, 
                                                  "The text shows complex emotional content.")
        
        return explanation
    
    def create_emotion_chart(self, emotion_data):
        """Create a bar chart for emotion probabilities"""
        # Sort emotions by probability
        sorted_emotions = sorted(emotion_data.items(), key=lambda x: x[1], reverse=True)
        emotion_names = [e[0] for e in sorted_emotions]
        emotion_probs = [e[1] for e in sorted_emotions]
        
        # Create color list
        colors = [get_emotion_color(e) for e in emotion_names]
        
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(emotion_names, emotion_probs, color=colors)
        ax.set_title('Emotion Probabilities', fontsize=14)
        ax.set_ylabel('Probability', fontsize=12)
        ax.set_ylim(0, 1)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        return fig

    def create_sentiment_chart(self, sentiment_data):
        """Create a bar chart for sentiment analysis"""
        sentiment_labels = ['Negative', 'Neutral', 'Positive', 'Compound']
        sentiment_values = [sentiment_data['neg'], sentiment_data['neu'], 
                            sentiment_data['pos'], sentiment_data['compound']]
        sentiment_colors = ['#DC143C', '#A9A9A9', '#228B22', '#1E90FF']
        
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(sentiment_labels, sentiment_values, color=sentiment_colors)
        ax.set_title('Sentiment Analysis', fontsize=14)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_ylim(-1, 1)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.tight_layout()
        
        return fig

    def create_wordcloud(self, word_importance, text):
        """Create a frequency bar chart instead of wordcloud"""
        if not word_importance:
            # Create empty figure if no words
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, "No significant emotional words found", 
                   ha='center', va='center', fontsize=12)
            ax.axis('off')
            return fig
        
        # Sort words by importance
        sorted_words = sorted(word_importance.items(), key=lambda x: x[1], reverse=True)[:15]
        words = [w[0] for w in sorted_words]
        importance = [w[1] for w in sorted_words]
        
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.barh(words, importance)
        ax.set_title('Important Words', fontsize=14)
        ax.set_xlabel('Importance', fontsize=12)
        ax.invert_yaxis()  # To have most important at the top
        plt.tight_layout()
        
        return fig

def get_emotion_color(emotion):
    """Get color for an emotion with better readability"""
    emotion_colors = {
        "neutral": "#D3D3D3",  # Light gray
        "joy": "#FFEAA7",      # Lighter yellow
        "sadness": "#A9D6F5",  # Lighter blue
        "fear": "#D8BFD8",     # Lighter purple
        "surprise": "#FFD580", # Lighter orange
        "anger": "#FFC0CB",    # Pink instead of dark red
        "shame": "#D2B48C",    # Lighter brown
        "disgust": "#98FB98",  # Lighter green
        "happy": "#FF91CB",    # pink
        "sad": "#103EBD",      # dark blue
        "angry": "#E74C3C",    # red
        "pleasant": "#F39C12", # orange
    }
    return emotion_colors.get(emotion, "#D3D3D3")

def set_page_config():
    """Configure the Streamlit page"""
    st.set_page_config(
        page_title="Emotion Analysis System",
        page_icon="😊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # CSS for full page light blue background
    st.markdown("""
    <style>
    /* Full page background */
    .stApp {
        background-color: #F0F4F8;
    }
    
    /* Make all containers transparent to show background */
    .css-1d391kg, .css-1lcbmhc, .css-18e3th9,
    .css-1r6slb0, .css-12oz5g7, .css-1oe6wy4,
    .css-6qob1r, .css-18ni7ap, .css-k1vhr4 {
        background-color: transparent !important;
    }
    
    /* Sidebar style */
    section[data-testid="stSidebar"] {
        background-color: #E2E8F0 !important;
        border-right: 1px solid #CBD5E0;
    }
    
    /* Header background */
    .header-style {
        background-color: #1E3A8A;
        padding: 1.5rem; 
        border-radius: 10px; 
        margin-bottom: 2rem; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Content containers */
    .content-box {
        background-color: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
    }
    
    /* Main text styles */
    .main-header {
        color: white; 
        text-align: center; 
        font-size: 2.5rem; 
        margin: 0;
    }
    
    .sub-header {
        color: #E0E7FF; 
        text-align: center; 
        margin-top: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

def login_page():
    """Display a standard login page with visually distinct input fields and header"""
    # Title with dark blue color
    st.markdown('<h1 style="color: #1E3A8A; text-align: center; font-size: 3.2rem; margin-bottom: 1.7rem;">Emotion Analysis System</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center; color:#64748B; font-size:1.1rem; margin-bottom:2rem;">Detect and analyze emotions in text and audio</p>', unsafe_allow_html=True)
    
    # Check if registration was successful
    if 'register_success' in st.session_state and st.session_state['register_success']:
        st.success("Account created successfully!")
        st.session_state['register_success'] = False
    
    # Check if we should show registration or login
    if 'show_register' not in st.session_state:
        st.session_state['show_register'] = False
    
    # Add custom CSS for visually distinct text fields with proper eye icon alignment
    st.markdown("""
    <style>
    /* Main container styling */
    .login-container {
        max-width: 360px;
        margin: 0 auto;
        padding: 20px;
    }
    
    /* Form title */
    .login-title {
        color: #1E3A8A;
        font-size: 22px;
        text-align: center;
        margin-bottom: 20px;
        font-weight: bold;
    }
    
    /* Emoji bar */
    .emoji-bar {
        font-size: 30px;
        text-align: center;
        margin-bottom: 20px;
    }
    
    /* Make input fields clearly visible */
    .stTextInput input {
        background-color: white !important;
        border: 1px solid #CBD5E0 !important;
        border-radius: 5px !important;
        padding: 10px 12px !important;
        color: #1F2937 !important;
        font-size: 15px !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
        height: 42px !important;
    }
    
    /* Add focus highlighting */
    .stTextInput input:focus {
        border-color: #3B82F6 !important;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.3) !important;
    }
    
    /* Fix for password field with eye icon */
    .stTextInput div[data-baseweb="input"] {
        max-width: 320px !important;
        margin: 0 auto !important;
        background-color: white !important;
        border-radius: 5px !important;
    }
    
    /* Password visibility toggle position fix */
    .stTextInput div[data-baseweb="input"] > div:last-child {
        position: absolute !important;
        right: 0 !important;
        height: 42px !important;
        display: flex !important;
        align-items: center !important;
        padding-right: 10px !important;
    }
    
    /* Make placeholder text visible */
    .stTextInput input::placeholder {
        color: #9CA3AF !important;
        opacity: 1 !important;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #1E3A8A !important;
        color: white !important;
        max-width: 320px !important;
        height: 42px !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        transition: all 0.2s !important;
        margin-top: 15px;
    }
    
    .stButton > button:hover {
        background-color: #1E40AF !important;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1) !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Simple emoji row with basic styling
    st.markdown('<div class="emoji-bar">😊 😢 😡 😲 😨 🤢 😳</div>', unsafe_allow_html=True)
    
    # Centered container
    col1, col2, col3 = st.columns([1, 1.5, 1])
    
    with col2:
        if not st.session_state['show_register']:
            # Login Form
            st.markdown('<div class="login-title">Sign In</div>', unsafe_allow_html=True)
            username = st.text_input("Username", key="login_username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", key="login_password", placeholder="Enter your password")
            
            col_btn1, col_btn2 = st.columns([1, 1], gap="small")
            with col_btn1:
                if st.button("Sign In", key="signin_button", use_container_width=True):
                    if username and password:
                        if authenticate_user(username, password):
                            st.session_state['logged_in'] = True
                            st.session_state['username'] = username
                            st.session_state['show_welcome'] = True
                            st.rerun()
                        else:
                            st.error("Invalid username or password")
                    else:
                        st.warning("Please enter credentials")
            
            with col_btn2:
                if st.button("Create Account", key="create_account_button", use_container_width=True):
                    st.session_state['show_register'] = True
                    st.rerun()
        
        else:
            # Registration Form
            st.markdown('<div class="login-title">Create Account</div>', unsafe_allow_html=True)
            new_username = st.text_input("Username", key="register_username", placeholder="Choose a username")
            new_password = st.text_input("Password", type="password", key="register_password", placeholder="Create a password")
            confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password", placeholder="Confirm your password")
            
            col_btn1, col_btn2 = st.columns([1, 1], gap="small")
            with col_btn1:
                if st.button("Register", key="register_button", use_container_width=True):
                    if new_username and new_password:
                        import re
                        if re.search(r'[^a-zA-Z0-9]', new_username):
                            st.error("Username must contain only letters & numbers")
                        elif new_username.isdigit():
                            st.error("Username must include letters")
                        elif new_password != confirm_password:
                            st.error("Passwords does not match")
                        elif len(new_password) < 6:
                            st.error("Password must be at least 6 characters long")
                        else:
                            if add_user(new_username, new_password):
                                st.session_state['register_success'] = True
                                st.session_state['show_register'] = False
                                st.rerun()
                            else:
                                st.error("Username already exists")
                    else:
                        st.warning("Please fill in all fields")
            
            with col_btn2:
                if st.button("Back to Login", key="back_button", use_container_width=True):
                    st.session_state['show_register'] = False
                    st.rerun()

def app_sidebar():
    """Display the sidebar navigation"""
    #if 'page' not in st.session_state:
     #  st.session_state.page = "Text Analysis"

    current_page = st.session_state.get("page", "Text Analysis")

    #subtitle_text = "Discover emotions in text"
    if 'current_page' in locals() and current_page == "Audio Analysis":
        subtitle_text = "Discover emotions in audio"
    elif 'current_page' in locals() and current_page == "History":
        subtitle_text = "History"
    elif 'current_page' in locals() and current_page == "About":
        subtitle_text = "About Us"
    else:
        subtitle_text = "Discover emotions in text"

    st.sidebar.markdown("""
    <style>
        .sidebar-title {
            text-align: center;
            padding: 1rem 0;
        }
        .sidebar-title h2 {
            color: #1E3A8A;
            margin-bottom: 0.5rem;
        }
        .sidebar-subtext {
            color: #64748B;
            font-size: 0.9rem;
            margin-top: 0;
            text-align: center;
        }
        .sidebar-divider {
            height: 1px;
            background-color: #E2E8F0;
            margin: 0.5rem 0 1.5rem 0;
        }
    </style>
    """, unsafe_allow_html=True)

    # Display title and styled subtitle in sidebar
    st.sidebar.markdown(f"""
    <div class="sidebar-title">
        <h2>Emotion Analysis</h2>
        <p class="sidebar-subtext">{subtitle_text}</p>
    </div>
    <div class="sidebar-divider"></div>
    """, unsafe_allow_html=True)
    
    # Add navigation with built-in Streamlit components
    st.sidebar.markdown("<h3 style='color: #475569; font-size: 1rem; margin-bottom: 1rem;'>Navigation</h3>", unsafe_allow_html=True)
    
    # Create navigation using Streamlit's radio component with custom styling
    page = st.sidebar.radio(
        label="",
        options=["Text Analysis", "Audio Analysis", "History", "About"],
        label_visibility="collapsed"
    )

    # Update page in session state if changed
    if page != current_page:
        st.session_state.page = page
        st.rerun()
    
    # Add user info card
    st.sidebar.markdown("<div style='height: 1px; background-color: #E2E8F0; margin: 1.5rem 0;'></div>", unsafe_allow_html=True)
    st.sidebar.markdown(f"""
    <div style="background-color: #F1F5F9; padding: 1rem; border-radius: 0.5rem; margin-top: 1rem;">
        <div style="display: flex; align-items: center;">
            <div style="background-color: #3B82F6; color: white; border-radius: 50%; width: 40px; height: 40px; display: flex; align-items: center; justify-content: center; font-size: 1.2rem; margin-right: 0.75rem;">
                {st.session_state['username'][0].upper() if st.session_state['username'] else 'U'}
            </div>
            <div>
                <p style="margin: 0; font-size: 0.9rem; color: #475569;">Logged in as:</p>
                <p style="margin: 0; font-weight: bold; color: #1E3A8A;">{st.session_state['username']}</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Add logout button with proper session state cleanup
    if st.sidebar.button("Logout", type="primary"):
        # Clear all relevant session states
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        # Initialize required session states
        st.session_state['logged_in'] = False
        st.session_state['username'] = None
        st.session_state['show_register'] = False
        st.rerun()
    
    # Add version information at the bottom
    st.sidebar.markdown("""
    <div style="text-align: center; padding-top: 2rem; color: #94A3B8; font-size: 0.8rem;">
        <p style="margin: 0;">Emotion Analysis System v2.0</p>
        <p style="margin: 0;">© 2025 Sandra & Hui Lin</p>
    </div>
    """, unsafe_allow_html=True)
    
    return page


# Text Part
def text_analysis_page(analyzer):
    # Show welcome message if needed
    if st.session_state.get('show_welcome', False):
        username = st.session_state.get('username', '')
        st.markdown(f"""
    <div id="welcome-toast" style="
        position: fixed;
        top: 70px;
        left: 50%;
        transform: translateX(-50%);
        background-color: #6DD19C;
        color: white;
        padding: 20px 30px;
        border-radius: 10px;
        z-index: 9999;
        text-align: center;
        box-shadow: 0 6px 16px rgba(0,0,0,0.3);
        animation: fadeInOut 5s forwards;
        max-width: 80%;
        width: 500px;
        margin: 0 auto;
    ">
        <h3 style="margin: 0; font-size: 22px; font-weight: bold;">👋 Welcome, {username}!</h3>
        <p style="margin: 8px 0 0 0; font-size: 16px;">Successfully logged in to Emotion Analysis System</p>
    </div>
    
    <style>
        @keyframes fadeInOut {{
            0% {{ opacity: 0; transform: translate(-50%, -30px); }}
            15% {{ opacity: 1; transform: translate(-50%, 0); }}
            85% {{ opacity: 1; transform: translate(-50%, 0); }}
            100% {{ opacity: 0; transform: translate(-50%, -30px); }}
        }}
        
        #welcome-toast {{
            display: block !important;
            visibility: visible !important;
        }}
    </style>
    """, unsafe_allow_html=True)
    
    # Clear the welcome flag
    st.session_state['show_welcome'] = False
    
    # Create a better header with styling
    st.markdown("""
    <div class="header-style">
        <h1 class="main-header">Text Emotion Analysis</h1>
        <p class="sub-header">Discover the emotional content in any text</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ADD THIS CSS BLOCK HERE
    st.markdown("""
    <style>
    /* Add visible border to text area */
    textarea {
        border: 1px solid #CBD5E0 !important;
        border-radius: 5px !important;
    }
    
    /* Change border color on focus */
    textarea:focus {
        border: 2px solid #3B82F6 !important;
        box-shadow: 0 0 0 1px rgba(59, 130, 246, 0.3) !important;
    }
    
    /* Analysis Results Highlight */
    .analysis-progress {
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create two columns for layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Text input with better styling
        st.markdown('<div class="content-box"><label style="font-size: 1.2rem; font-weight: bold; color: #1E3A8A;">Enter text to analyze:</label>', unsafe_allow_html=True)
        
        # Initialize text_input key in session_state if not present
        if 'text_input' not in st.session_state:
            st.session_state['text_input'] = ""
            
        # Track if analysis is in progress
        if 'analysis_in_progress' not in st.session_state:
            st.session_state['analysis_in_progress'] = False
            
        # Text input handling - Direct approach
        text_input = st.text_area(
            "", 
            height=180, 
            value=st.session_state['text_input'],
            placeholder="Type or paste your text here...",
            key="text_area_input"
        )
        
        # Update session state directly (no need for double entry)
        st.session_state['text_input'] = text_input
            
        # Create action buttons with improved styling
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
        
        with col_btn1:
            analyze_button = st.button(
                "Analyze Emotion",
                key="analyze_button",
                type="primary",
                use_container_width=True,
                disabled=st.session_state.get('analysis_in_progress', False)
            )
        
        with col_btn2:
            example_button = st.button(
                "Try an example",
                key="example_button",
                use_container_width=True,
                disabled=st.session_state.get('analysis_in_progress', False)
            )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Information card with emotion examples
        st.markdown("""
        <div class="content-box">
            <h3 style="color: #1E3A8A; font-size: 1.1rem; margin-bottom: 0.5rem;">Detectable Emotions</h3>
            <ul style="padding-left: 1.5rem; margin-bottom: 0;">
                <li style="margin-bottom: 0.3rem;"><span style="color: #FFD700;">😊 Joy</span></li>
                <li style="margin-bottom: 0.3rem;"><span style="color: #1E90FF;">😢 Sadness</span></li>
                <li style="margin-bottom: 0.3rem;"><span style="color: #DC143C;">😡 Anger</span></li>
                <li style="margin-bottom: 0.3rem;"><span style="color: #FFA500;">😲 Surprise</span></li>
                <li style="margin-bottom: 0.3rem;"><span style="color: #8A2BE2;">😨 Fear</span></li>
                <li style="margin-bottom: 0.3rem;"><span style="color: #228B22;">🤢 Disgust</span></li>
                <li style="margin-bottom: 0.3rem;"><span style="color: #8B4513;">😳 Shame</span></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Handle example button
    if example_button:
        examples = [
            "I'm so angry about you because you have broke my heart",
            "I'm incredibly happy today! Everything seems to be going well and I can't stop smiling.",
            "The news was so shocking, I couldn't believe what I was hearing.",
            "I'm really anxious about the upcoming presentation, I'm afraid I'll mess it up.",
            "I'm feeling a mix of excitement and nervousness about tomorrow's event.",
            "I can't believe they would do that to me after everything we've been through.",
            "The sunset today was absolutely breathtaking, I felt so peaceful and content."
        ]
        # Pick a random example
        import random
        text_input = random.choice(examples)
        st.session_state['text_input'] = text_input
        # Automatically analyze examples
        st.session_state['auto_analyze'] = True
        st.rerun()
    
    # Auto-analyze after selecting an example
    if 'auto_analyze' in st.session_state and st.session_state['auto_analyze']:
        analyze_button = True
        st.session_state['auto_analyze'] = False
    
    # Perform analysis when analyze button is clicked
    if analyze_button:
        # Check if the text field is empty
        if not text_input or text_input.strip() == "":
            st.error("The text field cannot be left empty. Please enter some text to analyze.")
        else:
            # Check if text contains at least one letter
            import re
            if not re.search('[a-zA-Z]', text_input):
                st.error("The text cannot contain only symbols and numbers. Please include some words for a valid analysis.")
            else:
                try:
                    # Set analysis in progress flag
                    st.session_state['analysis_in_progress'] = True
                    
                    # Create a progress bar for analysis steps
                    progress_bar = st.progress(0)
                    
                    # Step 1: Preprocessing
                    st.markdown('<div class="analysis-progress">Preprocessing text...</div>', unsafe_allow_html=True)
                    progress_bar.progress(20)
                    time.sleep(0.2)  # Small delay for visual feedback
                    
                    # Step 2: Analyzing
                    st.markdown('<div class="analysis-progress">Analyzing emotions...</div>', unsafe_allow_html=True)
                    progress_bar.progress(40)
                    time.sleep(0.3)  # Small delay for visual feedback
                    
                    # Perform analysis with text_input
                    analysis = analyzer.analyze_text(text_input)
                    
                    # Step 3: Processing results
                    st.markdown('<div class="analysis-progress">Processing results...</div>', unsafe_allow_html=True)
                    progress_bar.progress(70)
                    time.sleep(0.2)  # Small delay for visual feedback
                    
                    # Step 4: Generating visuals
                    st.markdown('<div class="analysis-progress">Generating visuals...</div>', unsafe_allow_html=True)
                    progress_bar.progress(90)
                    time.sleep(0.2)  # Small delay for visual feedback
                    
                    # Save to user history
                    history_entry = {
                        'text': text_input,
                        'emotion': analysis['predicted_emotion'],
                        'confidence': analysis['confidence'],
                        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    add_to_history(st.session_state['username'], history_entry)

                    # Complete progress
                    progress_bar.progress(100)
                    time.sleep(0.1)  # Small delay for visual feedback
                    
                    # Clear the progress indicators
                    progress_bar.empty()
                    
                    # Reset analysis flag
                    st.session_state['analysis_in_progress'] = False
                    
                    # Display the analysis results
                    display_analysis_results(analysis, analyzer, text_input)
                    
                    # Add a hidden success message that fades in/out
                    st.markdown("""
                    <div class="toast-notification">
                        Analysis complete! Results saved to history.
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"An error occurred during analysis: {str(e)}")
                    st.error("Please try again with different text or contact support.")
                    # Reset analysis flag
                    st.session_state['analysis_in_progress'] = False

def display_analysis_results(analysis, analyzer, text_input):
    """Separate function to display analysis results"""
    # Create nice card for results
    primary_emotion = analysis['predicted_emotion']
    mixed_emotion = analysis['mixed_with']
    emotion_color = get_emotion_color(primary_emotion)
    
    # Result container
    st.markdown("""
    <div class="content-box">
        <h2 style="color: #1E3A8A; margin-bottom: 1rem; border-bottom: 2px solid #E0E7FF; padding-bottom: 0.5rem;">Analysis Results</h2>
    """, unsafe_allow_html=True)
    
    # Main emotion result
    if mixed_emotion:
        mixed_color = get_emotion_color(mixed_emotion)
        st.markdown(
            f"""
            <div style="display: flex; margin-bottom: 1rem;">
                <div style="background-color: {emotion_color}; border-radius: 10px; border: 1px solid #E0E0E0; padding: 1rem; flex: 3; margin-right: 1rem; text-align: center;">
                    <h3 style="color: #000; margin: 0; font-size: 1.8rem;">{primary_emotion.upper()}</h3>
                    <p style="color: #333; margin: 0.5rem 0 0 0;">Primary Emotion</p>
                    <p style="font-size: 1.2rem; color: #333; margin: 0.5rem 0 0 0;">Confidence: {analysis['confidence']:.2f}</p>
                </div>
                <div style="background-color: {mixed_color}; border-radius: 10px; border: 1px solid #E0E0E0; padding: 1rem; flex: 2; text-align: center;">
                    <h3 style="color: #000; margin: 0; font-size: 1.5rem;">{mixed_emotion.upper()}</h3>
                    <p style="color: #333; margin: 0.5rem 0 0 0;">Secondary Emotion</p>
                </div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style="background-color: {emotion_color}; padding: 1.5rem; border-radius: 10px; border: 1px solid #E0E0E0; margin-bottom: 1rem; text-align: center;">
                <h3 style="color: #000; margin: 0; font-size: 2rem;">{primary_emotion.upper()}</h3>
                <p style="font-size: 1.2rem; color: #333; margin: 0.5rem 0 0 0;">Confidence: {analysis['confidence']:.2f}</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    # Explanation section
    explanation = analyzer.explain_analysis(analysis)
    st.markdown("<h3 style='color: #1E3A8A; margin-top: 1.5rem;'>Detailed Analysis</h3>", unsafe_allow_html=True)
    formatted_explanation = explanation.replace('\n', '<br>')
    st.markdown(f"<div style='background-color: #F8FAFC; padding: 1rem; border-radius: 5px; border-left: 4px solid #3B82F6;'>{formatted_explanation}</div>", unsafe_allow_html=True)
    
    # Keywords with colored badges
    st.markdown("<h3 style='color: #1E3A8A; margin-top: 1.5rem;'>Key Emotional Words</h3>", unsafe_allow_html=True)
    
    html_keywords = "<div style='margin: 0.5rem 0; line-height: 2.5;'>"
    for emotion, words in analysis['emotional_keywords'].items():
        if words:
            emotion_color = get_emotion_color(emotion)
            for word, weight in words[:3]:  # Top 3 words per emotion
                html_keywords += f'<span style="background-color: {emotion_color}; border: 1px solid #333; padding: 0.3rem 0.6rem; border-radius: 20px; margin-right: 0.5rem; font-weight: 500; color: #000;">{word} ({emotion})</span> '
    html_keywords += "</div>"
    
    st.markdown(html_keywords, unsafe_allow_html=True)
    
    # Close the container
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Visualizations section in a content box
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.markdown("<h2 style='color: #1E3A8A; margin: 0 0 1rem 0;'>Visualizations</h2>", unsafe_allow_html=True)
    
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        # Emotion chart
        st.markdown("<h3 style='color: #1E3A8A; font-size: 1.2rem;'>Emotion Distribution</h3>", unsafe_allow_html=True)
        emotion_chart = analyzer.create_emotion_chart(analysis['emotion_probabilities'])
        st.pyplot(emotion_chart)
        
        # Important words
        st.markdown("<h3 style='color: #1E3A8A; font-size: 1.2rem; margin-top: 1rem;'>Important Words</h3>", unsafe_allow_html=True)
        word_chart = analyzer.create_wordcloud(analysis['word_importance'], text_input)
        st.pyplot(word_chart)
        
    with viz_col2:
        # Sentiment chart
        st.markdown("<h3 style='color: #1E3A8A; font-size: 1.2rem;'>Sentiment Analysis</h3>", unsafe_allow_html=True)
        sentiment_chart = analyzer.create_sentiment_chart(analysis['sentiment_values'])
        st.pyplot(sentiment_chart)
        
        # Display input text with highlighted emotional words - IMPROVED VERSION
        st.markdown("<h3 style='color: #1E3A8A; font-size: 1.2rem; margin-top: 1rem;'>Highlighted Text</h3>", unsafe_allow_html=True)
        
        # Define high-emotion words (words that actually express emotion)
        high_emotion_words = {
            'anger': ['mad', 'angry', 'furious', 'outraged', 'enraged', 'hate', 'irritated', 'annoyed', 'frustrat', 'rage'],
            'joy': ['happy', 'delighted', 'thrilled', 'excited', 'wonderful', 'love', 'enjoyed', 'enjoy', 'glad', 'great'],
            'sadness': ['sad', 'upset', 'depressed', 'miserable', 'devastated', 'heartbroken', 'crying', 'lonely', 'hurt'],
            'fear': ['afraid', 'scared', 'terrified', 'frightened', 'worried', 'anxious', 'nervous', 'dread', 'panic'],
            'surprise': ['surprised', 'shocked', 'amazed', 'astonished', 'unexpected', 'wow', 'unbelievable'],
            'disgust': ['disgusted', 'gross', 'revolting', 'nasty', 'sick', 'yuck', 'ew', 'repulsed'],
            'shame': ['ashamed', 'embarrassed', 'guilty', 'regret', 'sorry', 'humiliated'],
            'neutral': [],
            # Add mappings for your model's specific categories
            'happy': ['happy', 'delighted', 'thrilled', 'excited', 'wonderful', 'love', 'enjoyed', 'enjoy', 'glad', 'great'],
            'sad': ['sad', 'upset', 'depressed', 'miserable', 'devastated', 'heartbroken', 'crying', 'lonely', 'hurt'],
            'angry': ['mad', 'angry', 'furious', 'outraged', 'enraged', 'hate', 'irritated', 'annoyed', 'frustrat', 'rage'],
            'pleasant': ['pleasant', 'nice', 'enjoyable', 'satisfying', 'pleasing', 'delightful', 'good', 'lovely']
        }
        
        # Get actual emotional words from text
        emotional_words = set()
        for emotion in high_emotion_words:
            for word in high_emotion_words[emotion]:
                if any(word in w.lower() for w in text_input.split()):
                    emotional_words.add(word)
        
        # Create a colored version of the input text
        words = text_input.split()
        highlighted_text = ""
        
        for word in words:
            word_clean = word.lower().strip(string.punctuation)
            colored = False
            
            # Check if this word contains any emotional word
            for emotion, emotion_word_list in high_emotion_words.items():
                if any(e_word in word_clean for e_word in emotion_word_list):
                    emotion_color = get_emotion_color(emotion)
                    highlighted_text += f'<span style="background-color: {emotion_color}; border: 1px solid #333; padding: 2px 4px; border-radius: 3px; color: #000;">{word}</span> '
                    colored = True
                    break
            
            # If not colored already and it's a significant word in the analysis
            if not colored and word_clean in analysis.get('word_importance', {}):
                # Only highlight if the word importance is significant
                if analysis['word_importance'][word_clean] > 0.5:
                    emotion_color = get_emotion_color(primary_emotion)
                    highlighted_text += f'<span style="background-color: {emotion_color}; border: 1px solid #333; padding: 2px 4px; border-radius: 3px; color: #000;">{word}</span> '
                    colored = True
            
            if not colored:
                highlighted_text += word + " "
        
        st.markdown(f'<div style="background-color: white; padding: 1rem; border-radius: 0.5rem; border: 1px solid #e0e0e0; line-height: 1.6;">{highlighted_text}</div>', unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)


# Audio Part
emotion_color = {
    "happy": "#FF91CB", #pink
    "sad": "#103EBD", #dark blue
    "angry": "#E74C3C", #red
    "fear" : "#8E44AD", #purple
    "pleasant": "#F39C12", #orange
    "disgust" : "#27AE60", #green
    "neutral" : "#95A5A6",   #grey
    "joy": "#FFEAA7",      # Lighter yellow
    "sadness": "#A9D6F5",  # Lighter blue 
    "anger": "#FFC0CB",    # Pink instead of dark red
    "surprise": "#FFD580", # Lighter orange
    "shame": "#D2B48C"     # Lighter brown
}

emotion_emoji = {
    "happy": "😄",
    "sad": "😢",
    "angry": "😠",
    "fear": "😨",
    "pleasant": "😊",
    "disgust": "🤢",
    "neutral": "😐",
    "joy": "😊",
    "sadness": "😢",
    "anger": "😡",
    "surprise": "😲",
    "shame": "😳"
}

def audio_analysis_page():
    """Display the audio analysis page"""
    st.markdown("""
    <div class="header-style">
        <h1 class="main-header">Audio Emotion Analysis</h1>
        <p class="sub-header">Detect emotional tone in audio recoding</p>
    </div>
    """, unsafe_allow_html=True)
    
    def log_error(e):
        st.error("An error occurred:")
        st.error(str(e))
        st.error("Detailed error trace:")
        st.error(traceback.format_exc())
    
    def check_files():
        project_path = "C:/Users/esther/OneDrive - Tunku Abdul Rahman University College/FYP/Final 2/Emotion Detection Text Sentiment Analysis Final"
        os.chdir(project_path)
        required_files = [
            os.path.join(project_path, 'audio_sentiment_model.keras'),
            os.path.join(project_path, 'encoder_label.pkl'),
            os.path.join(project_path, 'scaler_data.pkl')
        ]
        missing_files = []
        
        for file in required_files:
            if not os.path.exists(file):
                missing_files.append(file)
        
        return missing_files

    def add_noise(data):
        noise_value = 0.015 * np.random.uniform() * np.amax(data)
        data = data + noise_value * np.random.normal(size=data.shape[0])
        return data

    def stretch_process(data, rate=0.8):
        return librosa.effects.time_stretch(y=data, rate=rate)

    def pitch_process(data, sampling_rate, pitch_factor=0.7):
        return librosa.effects.pitch_shift(y=data, sr=sampling_rate, n_steps=pitch_factor)

    def extract_process(data, sample_rate):
        output_result = np.array([])

        mean_zero = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
        output_result = np.hstack((output_result, mean_zero))

        stft_out = np.abs(librosa.stft(data))
        chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft_out, sr=sample_rate).T, axis=0)
        output_result = np.hstack((output_result, chroma_stft))

        mfcc_out = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
        output_result = np.hstack((output_result, mfcc_out))

        root_mean_out = np.mean(librosa.feature.rms(y=data).T, axis=0)
        output_result = np.hstack((output_result, root_mean_out))

        mel_spectrogram = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
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
    
    def reshape_to_fixed_size(data, target_features=704):
        if data.shape[1] > target_features:
            return data[:, :target_features]
        elif data.shape[1] < target_features:
            pad_width = ((0, 0), (0, target_features - data.shape[1]))
            return np.pad(data, pad_width, mode='constant')
        return data
    
    def predict_emotion(audio_path):
        try:
            # Define data directories
            project_path = "C:/Users/esther/OneDrive - Tunku Abdul Rahman University College/FYP/Final 2/Emotion Detection Text Sentiment Analysis Final"
            if not os.path.exists(project_path):
                st.error(f"Project path not found: {project_path}")
                return None, None, None    
            
            # Load model and encoder
            if os.path.exists(os.path.join(project_path,'audio_sentiment_model.keras')):
                loaded_model = tf.keras.models.load_model(os.path.join(project_path,'audio_sentiment_model.keras'))
            else:
                st.error("Audio sentiment model not found. Please ensure files are in the correct location.")
                return None, None, None
                
            if os.path.exists(os.path.join(project_path,'encoder_label.pkl')):
                with open(os.path.join(project_path,'encoder_label.pkl'), 'rb') as f:
                    loaded_encoder = pickle.load(f)
            else:
                st.error("Encoder label file not found. Please ensure files are in the correct location.")
                return None, None, None

            # Load audio and extract features
            data, sample_rate = librosa.load(audio_path, duration=2.5, offset=0.6)
            features = export_process(data, sample_rate)

            # Load scaler and normalize features
            scaler_path = os.path.join(project_path, 'scaler_data.pkl')
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    scaler_data = pickle.load(f)
                features = scaler_data.transform(features)
            else:
                st.warning("Scaler file not found! Using a new scaler which may cause prediction errors.")
                scaler_data = StandardScaler()
                features = scaler_data.fit_transform(features)

            features = reshape_to_fixed_size(features)
            features = features.reshape(features.shape[0], features.shape[1], 1)

            # Make prediction
            predictions = loaded_model.predict(features)
            predicted_indices = np.argmax(predictions, axis=1)
            predicted_emotion = loaded_encoder.inverse_transform(predicted_indices)
            confidence_scores = np.max(predictions, axis=1)
            emotion = predicted_emotion[0]
            emotion_classes = loaded_encoder.classes_
            all_confidence_scores = predictions[0]
            
            return emotion, all_confidence_scores, emotion_classes
        
        except Exception as e:
            log_error(e)
            st.error(f"Prediction error: {str(e)}")
            return None, None, None
    
    # Add CSS for audio page
    st.markdown("""
    <style>
    .stFileUploader{
        background-color: #F8F9FA;
        border: 2px dashed #E9ECEF;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .emotion-text{
        font-size: 30px;
        font-weight: bold;
        color: var(--emotion-color);
        text-align: center;
        padding: 10px;
        border-radius: 10px;
        margin: 20px 0;
        background-color: rgba(var(--emotion-color-rgb), 0.1);
    }
    .emotion-emoji{
        font-size: 48px;
        text-align: center;
        display: block;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    #define project path
    project_path = "C:/Users/esther/OneDrive - Tunku Abdul Rahman University College/FYP/Final 2/Emotion Detection Text Sentiment Analysis Final"
    required_files = [
        os.path.join(project_path, 'audio_sentiment_model.keras'),
        os.path.join(project_path, 'encoder_label.pkl'),
        os.path.join(project_path, 'scaler_data.pkl')
    ]

    # Check for required model files
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        st.warning(f"Some model files are missing: {', '.join([os.path.basename(f) for f in missing_files])}")
        st.info("You need to place the audio model files in the main project folder:")
        st.code("""
        {project_path}/
        ├── audio_sentiment_model.keras
        ├── encoder_label.pkl
        └── scaler_data.pkl
        """)
        
        st.markdown("""
        ### Model Setup Instructions
        
        1. Copy the audio model files to the main project folder
        2. Make sure the filenames match exactly
        3. Restart the application
        """)
    else:
        # Audio uploader section
        st.markdown(f"""
        <h3 style="font-size: 2rem; font-weight: bold; color: #1E3A8A;">Upload Audio for Analysis</h3>
        """, unsafe_allow_html=True)     

        audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"], key = "unique_audio_uploader")
                
        # Process uploaded file if available
        if audio_file is not None:
            # Save uploaded file temporarily
            with open("temp_uploaded_audio.wav", "wb") as f:
                f.write(audio_file.getbuffer())
            
            # Display audio player
            st.audio(audio_file, format="audio/wav")
            
            # Analyze button for uploaded audio
            if st.button("Analyze Audio"):
                with st.spinner("Analyzing your audio file..."):
                    emotion, confidence_scores, emotion_classes = predict_emotion("temp_uploaded_audio.wav")
                    display_audio_results(emotion, confidence_scores, emotion_classes, emotion_color)
    
    st.markdown('</div>', unsafe_allow_html=True)

def hex_to_rgb(hex_color):
    """Convert hex color to RGB for CSS variable"""
    hex_color = hex_color.lstrip('#')
    return ','.join(str(int(hex_color[i:i+2], 16)) for i in (0,2,4))

def display_audio_results(emotion, confidence_scores, emotion_classes, emotion_color):
    """Helper function to display audio analysis results"""
    if emotion is not None:
        
        emotion = "neutral" if emotion.lower() == "neutral" else emotion.lower()
        color = emotion_color.get(emotion.lower(), "#000000")
        rgb_color = hex_to_rgb(color)
        emoji = emotion_emoji.get(emotion.lower(), "❓")

        #save audio analysis history
        try:
            confidence_value = float(np.max(confidence_scores)) if confidence_scores is not None else 0.0
            history_entry = {
                'text': f"[AUDIO ANALYSIS] Detected emotion: {emotion.upper()}",
                'emotion': emotion,
                'confidence': confidence_value,
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'is_audio': True,
                'emotion_color': color  
            }
            # Save to user history
            add_to_history(st.session_state['username'], history_entry)
            st.success("Analysis saved to history!")
        except Exception as e:
            st.warning(f"Could not save to history: {str(e)}")

        st.markdown(f"""
        <style>
        :root {{
            --emotion-color: {color};
            --emotion-color-rgb: {rgb_color};
        }}
        </style>
        <div class="emotion-emoji">{emoji}</div>
        <div class="emotion-text">Predicted Emotion: {emotion.upper()}</div>
        """, unsafe_allow_html=True)
        
        # Display emotion confidence chart
        if emotion_classes is not None and confidence_scores is not None:
            st.subheader("Emotion Confidence Scores")
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Create bars with colors
            colors = [emotion_color.get(emotion.lower(), "#666666") for emotion in emotion_classes]
            ax.bar(emotion_classes, confidence_scores, color=colors)
            
            # Customize chart
            ax.set_xlabel('Emotion')
            ax.set_ylabel('Confidence')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Display chart
            st.pyplot(fig)
            
            # Add a description of what the results mean
            st.markdown("""
            ### Understanding the Results
            
            The chart above shows the confidence scores for each emotion detected in your audio. 
            The higher the bar, the more confident the system is that this emotion is present.
            
            Audio emotion detection can be affected by:
            - Voice tone and pitch
            - Speaking speed
            - Background noise
            - Audio quality
            
            For best results, use clear audio recordings in a quiet environment.
            """)
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.error("Unable to analyze the audio. Please try a different file or check if the model files are correctly installed.")

# History
def history_page():
    """Display the history page"""
    st.markdown("""
    <div class="header-style">
        <h1 class="main-header">Analysis History</h1>
        <p class="sub-header">View your past emotion analyses</p>
    </div>
    """, unsafe_allow_html=True)
    
    history = get_history(st.session_state['username'])
    
    if not history:
        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        st.info("No analysis history found. Try analyzing some text first!")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.markdown(f"<h3 style='color: #1E3A8A; margin-bottom: 1rem;'>Your Recent Analysis</h3>", unsafe_allow_html=True)
    st.markdown(f"Showing your last {len(history)} analyses:")
    
    # Create a DataFrame for better display
    history_df = pd.DataFrame(history)
    history_df = history_df.sort_values(by='timestamp', ascending=False)
    
    # Display each history item
    for i, item in history_df.iterrows():
        emotion_color = get_emotion_color(item['emotion'])
        
        with st.expander(f"{item['timestamp']} - {item['emotion'].upper()} ({item['confidence']:.2f})"):
            st.markdown(f"""
            <div style="background-color: {emotion_color}; border-radius: 5px; padding: 0.5rem; margin-bottom: 0.5rem; display: inline-block;">
                <span style="font-weight: bold;">{item['emotion'].upper()}</span> ({item['confidence']:.2f})
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"**Text:** {item['text']}")
            
            # Show model type if available
            if 'model_type' in item:
                st.markdown(f"**Analysis method:** {item['model_type']} model")
            
            # Option to re-analyze
            if st.button(f"Re-analyze", key=f"reanalyze_{i}"):
                st.session_state['text_input'] = item['text']
                st.session_state['page'] = "Text Analysis"
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# About
def about_page():
    """Display the about page"""
    st.markdown("""
    <div class="header-style">
        <h1 class="main-header">About Emotion Analysis System</h1>
        <p class="sub-header">Understanding the technology behind emotion detection</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.markdown("""
    ### Overview
    This Emotion Analysis System uses advanced machine learning techniques to detect emotions in text. 
    The system recognizes multiple emotion categories including joy/happy, sadness/sad, anger/angry, fear, surprise, disgust, and pleasant.
    
    ### Features
    - **ML-Based Text Analysis**: Analyze emotional content in text with high accuracy using a trained deep learning model
    - **Mixed Emotion Detection**: Identifies complex emotional states with multiple feelings
    - **Audio Analysis**: Detect emotions in spoken language
    - **History**: Keep track of your previous analyses
    - **User Accounts**: Secure login system to save your history
    
    ### How it Works
    The system uses a deep learning approach for emotion detection:
    
    1. **Text Preprocessing**: Normalizes text by removing punctuation, converting to lowercase, etc.
    2. **Tokenization**: Converts text into numerical sequences that the model can understand
    3. **Deep Learning Model**: Uses a neural network trained on thousands of emotional text examples
    4. **Mixed Emotion Detection**: Identifies when multiple emotions are present in the same text
    5. **Visualization**: Provides intuitive charts and highlights to understand the emotional content
    
    ### Model Training
    The emotion detection model was trained on a large dataset of labeled emotional texts using:
    - Bidirectional LSTM layers for sequence understanding
    - Word embeddings to capture semantic relationships
    - Regularization techniques to prevent overfitting
    
    ### Privacy
    Your text inputs are processed locally and not shared externally.
    Only the emotion result and text are saved in your history.
    """)
    st.markdown('</div>', unsafe_allow_html=True)


def main():
    """Main application function"""
    set_page_config()
    init_db()
    
    # Initialize session state variables
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    if 'username' not in st.session_state:
        st.session_state['username'] = None
    if 'page' not in st.session_state:
        st.session_state['page'] = "Text Analysis"
    if 'text_input' not in st.session_state:
        st.session_state['text_input'] = ""
    if 'model_loaded' not in st.session_state:
        st.session_state['model_loaded'] = False
    if 'show_welcome' not in st.session_state:
        st.session_state['show_welcome'] = False
    if 'analysis_in_progress' not in st.session_state:
        st.session_state['analysis_in_progress'] = False
    
    # Check login status
    if not st.session_state['logged_in']:
        login_page()
    else:
        # Set the base path to your actual model location
        base_path = "C:/Users/esther/OneDrive - Tunku Abdul Rahman University College/FYP/Final 2/Emotion Detection Text Sentiment Analysis Final"
        
        # Construct full paths to model files
        model_path = f"{base_path}/emotion_detection_model/model.keras"
        tokenizer_path = f"{base_path}/emotion_detection_model/tokenizer.pickle"
        encoder_path = f"{base_path}/emotion_detection_model/label_encoder.pickle"
        max_len_path = f"{base_path}/emotion_detection_model/max_len.txt"
        
        # Only load the model once by checking session state
        if not st.session_state.get('emotion_analyzer'):
            try:
                # Display loading indicator
                with st.spinner("Loading emotion analysis system..."):
                    emotion_analyzer = EmotionAnalyzer(model_path, tokenizer_path, encoder_path, max_len_path)
                    if not emotion_analyzer.model_loaded:
                        # Create a fallback analyzer if needed
                        emotion_analyzer = EmotionAnalyzer(None, None, None, None)
                    # Store in session state so we don't reload every time
                    st.session_state['emotion_analyzer'] = emotion_analyzer
                    st.session_state['model_loaded'] = True
            except Exception as e:
                # Create a fallback analyzer silently
                emotion_analyzer = EmotionAnalyzer(None, None, None, None)
                st.session_state['emotion_analyzer'] = emotion_analyzer
        else:
            # Use the already loaded analyzer
            emotion_analyzer = st.session_state['emotion_analyzer']
        
        # Get page from sidebar
        page = app_sidebar()
        st.session_state['page'] = page
        
        # Display selected page
        if page == "Text Analysis":
            text_analysis_page(emotion_analyzer)
        elif page == "Audio Analysis":
            audio_analysis_page()
        elif page == "History":
            history_page()
        elif page == "About":
            about_page()


if __name__ == "__main__":
    main()