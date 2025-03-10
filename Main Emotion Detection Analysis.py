import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import datetime
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


# Simple Emotion Analyzer that doesn't require external dependencies
class SimpleEmotionAnalyzer:
    """Simple rule-based emotion analyzer"""
    def __init__(self):
        # Define emotion lexicons
        self.emotion_lexicon = {
            'joy': ['happy', 'joy', 'delighted', 'excited', 'pleased', 'glad', 'satisfied', 
                   'thrilled', 'elated', 'jubilant', 'love', 'enjoy', 'wonderful', 'amazing',
                   'fantastic', 'great', 'good', 'positive', 'beautiful', 'celebrate'],
            
            'sadness': ['sad', 'unhappy', 'depressed', 'miserable', 'heartbroken', 'gloomy',
                       'disappointed', 'hopeless', 'grief', 'sorrow', 'crying', 'tears', 
                       'tragedy', 'regret', 'miss', 'alone', 'lonely', 'despair', 'hurt', 'pain', 'broke'],
            
            'anger': ['angry', 'mad', 'furious', 'outraged', 'annoyed', 'irritated', 'enraged',
                     'frustrated', 'hate', 'hostile', 'rage', 'fury', 'hatred', 'resent', 
                     'disgusted', 'offended', 'bitter', 'aggressive', 'attack', 'violent'],
            
            'fear': ['afraid', 'scared', 'frightened', 'terrified', 'anxious', 'nervous', 'worried',
                    'panic', 'horror', 'terror', 'dread', 'alarmed', 'shock', 'apprehensive',
                    'threatened', 'uneasy', 'insecure', 'concern', 'doubt', 'stress'],
            
            'surprise': ['surprised', 'astonished', 'amazed', 'shocked', 'stunned', 'unexpected',
                        'startled', 'wonder', 'disbelief', 'bewildered', 'awe', 'speechless',
                        'sudden', 'incredible', 'unpredictable', 'remarkable', 'gasp', 'unbelievable'],
            
            'disgust': ['disgusted', 'revulsion', 'repulsed', 'aversion', 'nauseated', 'sickened',
                       'dislike', 'disapprove', 'offend', 'nasty', 'gross', 'foul', 'repulsive',
                       'repugnant', 'objectionable', 'offensive', 'vile', 'distaste', 'revolting', 'yuck'],
            
            'shame': ['ashamed', 'guilty', 'embarrassed', 'humiliated', 'shameful', 'regretful',
                     'sorry', 'remorse', 'fault', 'blame', 'disgraced', 'dishonor', 'mortified',
                     'conscience', 'apology', 'condemn', 'wrong', 'mistake', 'failure', 'disgrace'],
            
            'neutral': ['maybe', 'perhaps', 'possibly', 'probably', 'seems', 'appears', 'generally',
                       'typically', 'usually', 'occasionally', 'sometimes', 'often', 'regular', 
                       'normal', 'common', 'standard', 'average', 'moderate', 'medium', 'mild']
        }
        
        # Negation words that can flip emotion
        self.negation_words = ['not', 'no', 'never', 'neither', 'nor', "don't", "doesn't", 
                              "didn't", "won't", "wouldn't", "shouldn't", "couldn't", 
                              "can't", "isn't", "aren't", "wasn't", "weren't"]
    
    def preprocess_text(self, text):
        """Simple text preprocessing"""
        # Convert to lowercase
        text = str(text).lower()
        
        # Fix contractions manually
        for contraction, expansion in [
            ("don't", "do not"), ("doesn't", "does not"), ("didn't", "did not"),
            ("won't", "will not"), ("can't", "cannot"), ("i'm", "i am"),
            ("you're", "you are"), ("it's", "it is")
        ]:
            text = text.replace(contraction, expansion)
        
        # Split into tokens
        tokens = text.split()
        
        # Remove punctuation from tokens
        tokens = [token.strip(string.punctuation) for token in tokens]
        tokens = [token for token in tokens if token]  # Remove empty tokens
        
        return ' '.join(tokens), tokens
    
    def analyze_text(self, text):
        """Analyze the emotion content of text"""
        processed_text, tokens = self.preprocess_text(text)
        
        # Count emotions
        emotion_counts = {emotion: 0 for emotion in self.emotion_lexicon.keys()}
        contributing_words = {emotion: [] for emotion in self.emotion_lexicon.keys()}
        
        for i, token in enumerate(tokens):
            # Check if this is a negation
            negated = i > 0 and tokens[i-1] in self.negation_words
            
            # Check each emotion lexicon
            for emotion, words in self.emotion_lexicon.items():
                if token in words:
                    if not negated:
                        emotion_counts[emotion] += 1
                        contributing_words[emotion].append((token, 1))
            
            # Special case for anger about heartbreak
            if token == 'angry' and 'heart' in tokens and 'broke' in tokens:
                emotion_counts['anger'] += 2
                emotion_counts['sadness'] += 1
                contributing_words['anger'].append(('angry', 2))
                contributing_words['sadness'].append(('broke heart', 1))
        
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
        
        # Create word importance dictionary for visualization
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
        
        # Format emotional_keywords for display
        emotional_keywords = {}
        for emotion, words in contributing_words.items():
            if words:
                emotional_keywords[emotion] = words
        
        # Return the analysis results
        return {
            'text': text,
            'processed_text': processed_text,
            'predicted_emotion': primary_emotion,
            'mixed_with': mixed_emotion,
            'confidence': primary_score,
            'emotion_scores': emotion_counts,
            'emotion_probabilities': emotion_probs,
            'sentiment_values': sentiment,
            'emotional_keywords': emotional_keywords,
            'contributing_words': contributing_words,
            'word_importance': word_importance
        }
    
    def explain_analysis(self, analysis):
        """Create a human-readable explanation of the analysis"""
        if analysis['mixed_with']:
            explanation = f"Primary Emotion: {analysis['predicted_emotion'].upper()} mixed with {analysis['mixed_with'].upper()}\n"
            explanation += f"Confidence: {analysis['confidence']:.2f}\n\n"
        else:
            explanation = f"Primary Emotion: {analysis['predicted_emotion'].upper()}\n"
            explanation += f"Confidence: {analysis['confidence']:.2f}\n\n"
        
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
        predicted_emotion = analysis['predicted_emotion']
        mixed_emotion = analysis['mixed_with']
        
        # Customized explanation based on emotion
        emotion_explanations = {
            'joy': "The text expresses positive sentiments and happiness.",
            'sadness': "The text conveys feelings of sadness or melancholy.",
            'anger': "The text shows signs of frustration or anger.",
            'fear': "The text indicates anxiety or fearfulness.",
            'surprise': "The text expresses astonishment or unexpected reactions.",
            'disgust': "The text conveys feelings of aversion or repulsion.",
            'shame': "The text indicates embarrassment or shame.",
            'neutral': "The text appears to be fairly neutral in emotional content."
        }
        
        # Mixed emotion explanation
        if mixed_emotion:
            if predicted_emotion == 'anger' and mixed_emotion == 'sadness':
                explanation += "The text expresses anger mixed with sadness, suggesting feelings of hurt or betrayal."
            elif predicted_emotion == 'sadness' and mixed_emotion == 'anger':
                explanation += "The text expresses sadness with underlying anger, suggesting disappointment or feeling let down."
            else:
                explanation += f"The text primarily shows {predicted_emotion} with elements of {mixed_emotion}."
        else:
            explanation += emotion_explanations.get(predicted_emotion, 
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
        "disgust": "#98FB98"   # Lighter green
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
    """Display the login page with simpler styling - just border lines, no shadows"""
    # Title with dark blue color
    st.markdown('<h1 style="color: #1E3A8A; text-align: center; font-size: 3rem; margin-bottom: 1.5rem;">Emotion Analysis System</h1>', unsafe_allow_html=True)
    
    # Simple emoji row with basic styling
    st.markdown('<div style="text-align:center; font-size:40px; margin:20px 0;">😊 😢 😡 😲 😨 🤢 😳</div>', unsafe_allow_html=True)
    
    # Custom CSS for form elements - simplified with just borders, no shadows
    st.markdown("""
    <style>
    /* Clean input field styling with just border lines */
    input[type="text"], input[type="password"] {
        border: 1px solid #CBD5E0 !important;
        border-radius: 5px !important;
        padding: 8px !important;
        background-color: white !important;
        width: 100% !important;
        margin-bottom: 15px !important;
        font-size: 16px !important;
        box-shadow: none !important;
    }
    
    /* Input field focus state */
    input[type="text"]:focus, input[type="password"]:focus {
        border: 1px solid #3B82F6 !important;
        outline: none !important;
    }
    
    /* Label styling */
    .stTextInput > label {
        color: #1E3A8A !important;
        font-weight: bold !important;
        font-size: 16px !important;
        margin-bottom: 5px !important;
    }
    
    /* Form styling - clean, no shadows */
    [data-testid="stForm"] {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        max-width: 500px;
        margin: 0 auto;
        border: 1px solid #E2E8F0;
    }
    
    /* Button styling - cleaner look */
    .stButton > button {
        background-color: #1E3A8A !important;
        color: white !important;
        font-weight: bold !important;
        padding: 10px 20px !important;
        border-radius: 5px !important;
        border: none !important;
        cursor: pointer !important;
        transition: all 0.2s !important;
    }
    
    .stButton > button:hover {
        background-color: #2563EB !important;
    }
    
    /* Form title */
    .form-title {
        color: #1E3A8A;
        font-size: 24px;
        margin-bottom: 20px;
        text-align: center;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Check if we should show the registration form or login form
    if 'show_register' not in st.session_state:
        st.session_state['show_register'] = False
    
    if not st.session_state['show_register']:
        # Display login form in columns to control layout
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            # Add an explicit form title
            st.markdown('<p class="form-title">Sign In</p>', unsafe_allow_html=True)
            
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                col_a, col_b = st.columns(2)
                with col_a:
                    submit_button = st.form_submit_button("Sign In")
                with col_b:
                    register_button = st.form_submit_button("Create Account")
                
                if submit_button:
                    if username and password:
                        if authenticate_user(username, password):
                            st.session_state['logged_in'] = True
                            st.session_state['username'] = username
                            st.success("Login successful!")
                            st.rerun()
                        else:
                            st.error("Invalid username or password")
                    else:
                        st.warning("Please enter your username and password")
                
                if register_button:
                    st.session_state['show_register'] = True
                    st.rerun()
    else:
        # Display registration form in columns
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            # Add an explicit form title
            st.markdown('<p class="form-title">Create Account</p>', unsafe_allow_html=True)
            
            with st.form("register_form"):
                new_username = st.text_input("New Username")
                new_password = st.text_input("New Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    register_button = st.form_submit_button("Create Account")
                with col_b:
                    back_button = st.form_submit_button("Back to Login")
                
                if register_button:
                    if new_username and new_password:
                        if new_password != confirm_password:
                            st.error("Passwords do not match")
                        elif len(new_password) < 6:
                            st.error("Password must be at least 6 characters long")
                        else:
                            if add_user(new_username, new_password):
                                st.success("Account created successfully! You can now sign in.")
                                # Return to login page after a successful registration
                                st.session_state['show_register'] = False
                                st.rerun()
                            else:
                                st.error("Username already exists")
                    else:
                        st.warning("Please fill all fields")
                
                if back_button:
                    st.session_state['show_register'] = False
                    st.rerun()

def app_sidebar():
    """Display the sidebar navigation"""
    # Add an app title and logo to the sidebar
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h2 style="color: #1E3A8A; margin-bottom: 0.5rem;">Emotion Analysis</h2>
        <p style="color: #64748B; font-size: 0.9rem; margin-top: 0;">Discover emotions in text</p>
    </div>
    <div style="height: 1px; background-color: #E2E8F0; margin: 0.5rem 0 1.5rem 0;"></div>
    """, unsafe_allow_html=True)
    
    # Add navigation with built-in Streamlit components
    st.sidebar.markdown("<h3 style='color: #475569; font-size: 1rem; margin-bottom: 1rem;'>Navigation</h3>", unsafe_allow_html=True)
    
    # Create navigation using Streamlit's radio component with custom styling
    page = st.sidebar.radio(
        label="",
        options=["Text Analysis", "Audio Analysis", "History", "About"],
        label_visibility="collapsed"
    )
    
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
    
    # Add logout button
    if st.sidebar.button("Logout", type="primary"):
        st.session_state['logged_in'] = False
        st.session_state['username'] = None
        st.rerun()
    
    # Add version information at the bottom
    st.sidebar.markdown("""
    <div style="text-align: center; padding-top: 2rem; color: #94A3B8; font-size: 0.8rem;">
        <p style="margin: 0;">Emotion Analysis System v1.0</p>
        <p style="margin: 0;">© 2025 Sandra & Hui Lin</p>
    </div>
    """, unsafe_allow_html=True)
    
    return page


#Text Part
def text_analysis_page(analyzer):
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
    </style>
    """, unsafe_allow_html=True)
    
    # Create two columns for layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Text input with better styling
        st.markdown('<div class="content-box"><label style="font-size: 1.2rem; font-weight: bold; color: #1E3A8A;">Enter text to analyze:</label>', unsafe_allow_html=True)
        
        # Text input handling - KEY CHANGE: separate key for session state and widget
        text_input = st.text_area(
            "", 
            height=180, 
            value=st.session_state.get('text_input', ''),
            placeholder="Type or paste your text here...",
            key="text_area_widget"  # This key is just for the widget
        )
        
        # Save input text to session state immediately
        if text_input:
            st.session_state['text_input'] = text_input
        
        # Debug info
        # st.write(f"Current text: '{text_input}'")
        # st.write(f"Session state text: '{st.session_state.get('text_input', '')}'")
            
        # Create action buttons with improved styling
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
        
        with col_btn1:
            analyze_button = st.button(
                "Analyze Emotion",
                key="analyze_button",
                type="primary",
                use_container_width=True
            )
        
        with col_btn2:
            example_button = st.button(
                "Try an example",
                key="example_button",
                use_container_width=True
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
            "I'm really anxious about the upcoming presentation, I'm afraid I'll mess it up."
        ]
        # Pick a random example
        import random
        text_input = random.choice(examples)
        st.session_state['text_input'] = text_input
        st.experimental_rerun()  # Force a rerun to update the text area with the example
    
    # Now text_input will always have the current text
    current_text = st.session_state.get('text_input', '')
    
    # Perform analysis when analyze button is clicked and we have text
    if analyze_button and current_text:
        try:
            with st.spinner("Analyzing..."):
                # Perform analysis with current_text
                analysis = analyzer.analyze_text(current_text)
                
                # Save to user history
                history_entry = {
                    'text': current_text,
                    'emotion': analysis['predicted_emotion'],
                    'confidence': analysis['confidence'],
                    'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                add_to_history(st.session_state['username'], history_entry)
                
                # Display the analysis results
                display_analysis_results(analysis, analyzer, current_text)
                
        except Exception as e:
            st.error(f"An error occurred during analysis: {str(e)}")
            st.error("Please try again with different text or contact support.")

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
        
        # Display input text with highlighted emotional words
        st.markdown("<h3 style='color: #1E3A8A; font-size: 1.2rem; margin-top: 1rem;'>Highlighted Text</h3>", unsafe_allow_html=True)
        
        # Create a colored version of the input text
        words = text_input.split()
        highlighted_text = ""
        
        for word in words:
            word_clean = word.lower().strip(string.punctuation)
            colored = False
            
            for emotion, words_list in analysis['emotional_keywords'].items():
                if any(word_clean == w for w, _ in words_list):
                    emotion_color = get_emotion_color(emotion)
                    highlighted_text += f'<span style="background-color: {emotion_color}; border: 1px solid #333; padding: 2px 4px; border-radius: 3px; color: #000;">{word}</span> '
                    colored = True
                    break
            
            if not colored:
                highlighted_text += word + " "
        
        st.markdown(f'<div style="background-color: white; padding: 1rem; border-radius: 0.5rem; border: 1px solid #e0e0e0; line-height: 1.6;">{highlighted_text}</div>', unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)


#Audio Part
emotion_color = {
    "happy": "#FF91CB", #pink
    "sad": "#103EBD", #dark blue
    "angry": "#E74C3C", #red
    "fear" : "#8E44AD", #purple
    "pleasant": "#F39C12", #orange
    "disgust" : "#27AE60", #green
    "nuetral" : "#95A5A6"   #grey 
}

emotion_emoji = {
    "happy": "😄",
    "sad": "😢",
    "angry": "😠",
    "fear": "😨",
    "pleasant": "😊",
    "disgust": "🤢",
    "neutral": "😐"
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
        project_path = "C:/Users/esther/Downloads/Emotion Detection Text Sentiment Analysis"
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
            project_path = "C:/Users/esther/Downloads/Emotion Detection Text Sentiment Analysis"
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
    project_path = "C:/Users/esther/Downloads/Emotion Detection Text Sentiment Analysis"
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
            
            # Option to re-analyze
            if st.button(f"Re-analyze", key=f"reanalyze_{i}"):
                st.session_state['text_input'] = item['text']
                st.session_state['page'] = "Text Analysis"
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


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
    This Emotion Analysis System uses natural language processing techniques to detect emotions in text. 
    The system recognizes 8 emotion categories: joy, sadness, anger, fear, surprise, disgust, shame, and neutral.
    
    ### Features
    - **Text Analysis**: Analyze emotional content in text with high accuracy
    - **Mixed Emotion Detection**: Identifies complex emotional states with multiple feelings
    - **Audio Analysis**: (Coming soon) Detect emotions in spoken language
    - **History**: Keep track of your previous analyses
    - **User Accounts**: Secure login system to save your history
    
    ### How it Works
    The system uses a lexicon-based approach combined with contextual analysis:
    
    1. **Text Preprocessing**: Normalizes text by removing punctuation, converting to lowercase, etc.
    2. **Emotion Lexicon**: Uses a carefully curated dictionary of words associated with different emotions
    3. **Contextual Analysis**: Considers word relationships, negations, and intensifiers
    4. **Mixed Emotion Detection**: Identifies when multiple emotions are present in the same text
    5. **Visualization**: Provides intuitive charts and highlights to understand the emotional content
    
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
    
    # Check login status
    if not st.session_state['logged_in']:
        login_page()
    else:
        # Initialize analyzer
        analyzer = SimpleEmotionAnalyzer()
        
        # Get page from sidebar
        page = app_sidebar()
        st.session_state['page'] = page
        
        # Display selected page
        if page == "Text Analysis":
            text_analysis_page(analyzer)
        elif page == "Audio Analysis":
            audio_analysis_page()
        elif page == "History":
            history_page()
        elif page == "About":
            about_page()


if __name__ == "__main__":
    main()