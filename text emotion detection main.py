import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import sqlite3
import hashlib
import smtplib
import secrets
from email.message import EmailMessage
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = ""

# Database connection
def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    #c.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT, email TEXT, password TEXT)''')
    #c.execute('''CREATE TABLE IF NOT EXISTS reset_tokens (email TEXT, token TEXT)''')
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def add_user(username, email, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("INSERT INTO users VALUES (?, ?, ?)", (username, email, hash_password(password)))
    conn.commit()
    conn.close()

def login_user(username, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, hash_password(password)))
    user = c.fetchone()
    conn.close()
    return user

def send_reset_email(email, token):
    sender_email = "your_email@gmail.com"
    sender_password = "your_email_password"
    subject = "Password Reset Request"
    body = f"Click the link to reset your password: http://localhost:8501/?page=reset&token={token}"
    
    msg = EmailMessage()
    msg.set_content(body)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = email
    
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.send_message(msg)
        return True
    except Exception as e:
        print("Email error:", e)
        return False

def generate_reset_token(email):
    token = secrets.token_hex(16)
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("INSERT INTO reset_tokens VALUES (?, ?)", (email, token))
    conn.commit()
    conn.close()
    return token

def reset_password(token, new_password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT email FROM reset_tokens WHERE token=?", (token,))
    result = c.fetchone()
    if result:
        email = result[0]
        hashed_password = hash_password(new_password)
        c.execute("UPDATE users SET password=? WHERE email=?", (hashed_password, email))
        c.execute("DELETE FROM reset_tokens WHERE email=?", (email,))
        conn.commit()
    conn.close()

# Main Streamlit App
st.title("Emotion Detection in Text")

if not st.session_state.authenticated:
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type='password')
    if st.button("Login"):
        if login_user(username, password):
            st.session_state.authenticated = True
            st.session_state.username = username
            st.success(f"Welcome {username}!")
            st.rerun()

        else:
            st.error("Invalid Credentials")
    if st.button("Create Account"):
        st.session_state.authenticated = "signup"
        st.rerun()

else:
    if st.session_state.authenticated == "signup":
        st.subheader("Create Account")
        new_user = st.text_input("Username")
        new_email = st.text_input("Email")
        new_password = st.text_input("Password", type='password')
        if st.button("Sign Up"):
            add_user(new_user, new_email, new_password)
            st.session_state.authenticated = False
            st.success("Account created! Please log in.")
            st.rerun()

    else:
        st.subheader("Emotion Detection")
        user_text = st.text_area("Enter text to analyze:")
        if st.button("Analyze"):
            if user_text:
                def preprocess_text(text):
                    stop_words = set(stopwords.words('english'))
                    lemmatizer = WordNetLemmatizer()
                    text = text.lower()
                    text = re.sub(r'[^a-zA-Z\s]', '', text)
                    words = text.split()
                    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
                    return ' '.join(words)
                
                processed_text = preprocess_text(user_text)
                st.write("Processed Input:", processed_text)
                st.write("Emotion Detected: **Happy** (Example Output)")

# Initialize the database
init_db()

