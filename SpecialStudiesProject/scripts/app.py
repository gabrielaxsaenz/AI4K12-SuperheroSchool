import os
import json
import requests
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Securely load the OpenAI API key from an environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

API_URL = "https://api.openai.com/v1/completions"

def train_classifier(training_data, labels, reasons):
    """Train a Naive Bayes classifier on user choices."""
    
    if len(training_data) != len(labels):
        print(f"Error: Training data and labels have mismatched sizes: {len(training_data)} vs {len(labels)}")
        return None, None  

    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(training_data)
    classifier = MultinomialNB()
    classifier.fit(X_train, labels)
    
    return vectorizer, classifier

def explain_prediction(vectorizer, classifier, scenario):
    """Predicts the user's choice for the final scenario."""
    X_test = vectorizer.transform([scenario])
    
    # Get the probability of each class
    word_probs = classifier.predict_proba(X_test)[0]
    top_prediction = classifier.classes_[np.argmax(word_probs)]  # Get the highest probability choice
    
    return top_prediction  # Return the predicted choice

def generate_reasoning(scenario, choice, past_reasons):
    """Generates reasoning using OpenAI's GPT-4 API"""

    if not OPENAI_API_KEY:
        print("⚠️ Warning: OpenAI API key is missing. Using a default response.")
        return f"Option {choice} seems to align with your previous decision-making pattern."

    prompt = (
        f"Based on the user's previous decisions and reasoning, generate a thoughtful explanation "
        f"for why they might choose option {choice} in the following scenario:\n{scenario}\n\n"
        "User's past reasoning:\n" + "\n".join(past_reasons)
    )

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-4-turbo",
        "prompt": prompt,
        "max_tokens": 150,
        "temperature": 0.7
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=10)
        response.raise_for_status()  # Raise an error if the API response is not 200 OK
        result = response.json()
        return result["choices"][0]["text"].strip()

    except (requests.ConnectionError, requests.Timeout):
        print("⚠️ Warning: OpenAI API is unavailable. Using a default response.")
        return f"Option {choice} seems to align with your previous decision-making pattern."

    except requests.RequestException as e:
        print(f"⚠️ OpenAI API Error: {e}. Using a default response.")
        return f"Given your past choices, option {choice} appears to be the most logical decision."
