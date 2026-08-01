import streamlit as st
import tensorflow as tf
import numpy as np
import json
import pickle
import random
import nltk

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")

# Load files
model = tf.keras.models.load_model("chatbot_model.h5")

with open("intents.json") as file:
    intents = json.load(file)

words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

# ---------- Functions ----------

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [
        lemmatizer.lemmatize(word.lower()) for word in sentence_words
    ]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)

    for sw in sentence_words:
        for i, word in enumerate(words):
            if word == sw:
                bag[i] = 1

    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)

    res = model.predict(np.array([bow]), verbose=0)[0]

    ERROR_THRESHOLD = 0.25

    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)

    return_list = []

    for r in results:
        return_list.append(
            {"intent": classes[r[0]], "probability": str(r[1])}
        )

    return return_list


def get_response(intents_list):

    if len(intents_list) == 0:
        return "Sorry, I didn't understand that."

    tag = intents_list[0]["intent"]

    for intent in intents["intents"]:

        if intent["tag"] == tag:
            return random.choice(intent["responses"])

    return "Sorry, I couldn't find an answer."

# ---------- Streamlit UI ----------

st.set_page_config(
    page_title="SupportAI",
    page_icon="🤖"
)

with st.sidebar:

    st.title("🤖 SupportAI")

    st.write("### Customer Service Chatbot")

    st.info("""
This chatbot can help you with:

✅ Order Tracking

✅ Refunds

✅ Delivery

✅ Payments

✅ Login Issues

✅ Complaints
""")

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# User input
prompt = st.chat_input("Type your message...")

if prompt:

    # Show user message
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

   if len(st.session_state.messages) == 0:

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content":
            "👋 Hello! Welcome to SupportAI.\n\nHow can I help you today?"
        }
    )
