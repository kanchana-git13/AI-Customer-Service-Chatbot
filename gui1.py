import tkinter as tk
from tkinter import scrolledtext, messagebox, filedialog
import json
import random
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from datetime import datetime

# Initialize Lemmatizer
lemmatizer = WordNetLemmatizer()

# -------------------------------
# Load Chatbot Files
# -------------------------------
try:
    intents = json.loads(open("intents.json").read())
    words = pickle.load(open("words.pkl", "rb"))
    classes = pickle.load(open("classes.pkl", "rb"))
    model = load_model("chatbot_model.h5")
except Exception as e:
    messagebox.showerror("Error", f"Unable to load chatbot files.\n\n{e}")
    exit()

ERROR_THRESHOLD = 0.25

# -------------------------------
# Clean User Sentence
# -------------------------------
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [
        lemmatizer.lemmatize(word.lower())
        for word in sentence_words
    ]
    return sentence_words

# -------------------------------
# Bag of Words
# -------------------------------
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)

    bag = [0] * len(words)

    for sw in sentence_words:
        for i, w in enumerate(words):
            if w == sw:
                bag[i] = 1

    return np.array(bag)

# -------------------------------
# Predict Intent
# -------------------------------
def predict_class(sentence):
    bow = bag_of_words(sentence)

    res = model.predict(np.array([bow]), verbose=0)[0]

    results = [
        [i, r]
        for i, r in enumerate(res)
        if r > ERROR_THRESHOLD
    ]

    results.sort(key=lambda x: x[1], reverse=True)

    return_list = []

    for r in results:
        return_list.append(
            {
                "intent": classes[r[0]],
                "probability": str(r[1])
            }
        )

    return return_list

# -------------------------------
# Get Bot Response
# -------------------------------
def get_response(intents_list, intents_json):

    if len(intents_list) == 0:
        return "Sorry, I didn't understand that. Could you please rephrase?"

    tag = intents_list[0]["intent"]

    list_of_intents = intents_json["intents"]

    for intent in list_of_intents:

        if intent["tag"] == tag:
            return random.choice(intent["responses"])

    return "I'm sorry, I couldn't process your request."

# -------------------------------
# Main Chatbot Function
# -------------------------------
def chatbot_response(message):

    intents_list = predict_class(message)

    response = get_response(intents_list, intents)

    return response
# ==========================================
# CREATE MAIN WINDOW
# ==========================================

root = tk.Tk()
root.title("AI Customer Service Chatbot")
root.geometry("800x700")
root.configure(bg="#EAF2F8")
root.resizable(False, False)

# ==========================================
# TITLE
# ==========================================

title = tk.Label(
    root,
    text="🤖 AI Customer Service Chatbot",
    font=("Arial", 18, "bold"),
    bg="#1F618D",
    fg="white",
    pady=12
)
title.pack(fill=tk.X)

# ==========================================
# CHAT HISTORY FRAME
# ==========================================

chat_frame = tk.Frame(root, bg="#EAF2F8")
chat_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

scrollbar = tk.Scrollbar(chat_frame)

chat_history = tk.Text(
    chat_frame,
    wrap=tk.WORD,
    font=("Calibri", 12),
    bg="white",
    fg="black",
    yscrollcommand=scrollbar.set,
    state=tk.DISABLED
)

scrollbar.config(command=chat_history.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

chat_history.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# ==========================================
# WELCOME MESSAGE
# ==========================================

chat_history.config(state=tk.NORMAL)

chat_history.insert(
    tk.END,
    "Bot : Hello! 👋 Welcome to AI Customer Service Chatbot.\n"
)

chat_history.insert(
    tk.END,
    "Bot : How may I help you today?\n\n"
)

chat_history.config(state=tk.DISABLED)

# ==========================================
# INPUT FRAME
# ==========================================

input_frame = tk.Frame(root, bg="#EAF2F8")
input_frame.pack(fill=tk.X, padx=10, pady=5)

user_input = tk.Entry(
    input_frame,
    font=("Calibri", 13),
    width=60
)

user_input.pack(
    side=tk.LEFT,
    padx=5,
    pady=5,
    fill=tk.X,
    expand=True
)

# ==========================================
# BUTTON FRAME
# ==========================================

button_frame = tk.Frame(root, bg="#EAF2F8")
button_frame.pack(fill=tk.X, padx=10, pady=10)

# Send Button
send_button = tk.Button(
    button_frame,
    text="Send",
    width=12,
    bg="#2E86C1",
    fg="white",
    font=("Arial", 11, "bold")
)

send_button.grid(row=0, column=0, padx=8)

# Clear Button
clear_button = tk.Button(
    button_frame,
    text="Clear Chat",
    width=12,
    bg="#E67E22",
    fg="white",
    font=("Arial", 11, "bold")
)

clear_button.grid(row=0, column=1, padx=8)

# Save Button
save_button = tk.Button(
    button_frame,
    text="Save Chat",
    width=12,
    bg="#27AE60",
    fg="white",
    font=("Arial", 11, "bold")
)

save_button.grid(row=0, column=2, padx=8)

# Help Button
help_button = tk.Button(
    button_frame,
    text="Help",
    width=12,
    bg="#8E44AD",
    fg="white",
    font=("Arial", 11, "bold")
)

help_button.grid(row=0, column=3, padx=8)
# ==========================================
# UPDATE CHAT HISTORY
# ==========================================

def update_chat(sender, message):

    current_time = datetime.now().strftime("%I:%M %p")

    chat_history.config(state=tk.NORMAL)

    chat_history.insert(
        tk.END,
        f"[{current_time}] {sender}: {message}\n"
    )

    chat_history.config(state=tk.DISABLED)

    chat_history.yview(tk.END)


# ==========================================
# SEND MESSAGE
# ==========================================

def send_message(event=None):

    message = user_input.get().strip()

    if message == "":
        return

    update_chat("You", message)

    user_input.delete(0, tk.END)

    root.update()

    response = chatbot_response(message)

    update_chat("Bot", response)


# ==========================================
# CLEAR CHAT
# ==========================================

def clear_chat():

    answer = messagebox.askyesno(
        "Clear Chat",
        "Do you want to clear the conversation?"
    )

    if answer:

        chat_history.config(state=tk.NORMAL)

        chat_history.delete(1.0, tk.END)

        chat_history.insert(
            tk.END,
            "Bot : Hello! 👋 Welcome to AI Customer Service Chatbot.\n"
        )

        chat_history.insert(
            tk.END,
            "Bot : How may I help you today?\n\n"
        )

        chat_history.config(state=tk.DISABLED)


# ==========================================
# SAVE CHAT
# ==========================================

def save_chat():

    file = filedialog.asksaveasfilename(
        defaultextension=".txt",
        filetypes=[("Text File", "*.txt")]
    )

    if file:

        data = chat_history.get(1.0, tk.END)

        with open(file, "w", encoding="utf-8") as f:
            f.write(data)

        messagebox.showinfo(
            "Saved",
            "Chat history saved successfully."
        )


# ==========================================
# HELP WINDOW
# ==========================================

def help_window():

    messagebox.showinfo(
        "Help",

"""
AI Customer Service Chatbot

You can ask questions like:

• Hi
• Hello
• Track my order
• Cancel my order
• Refund
• Payment failed
• Shipping charges
• Delivery time
• Change address
• Login issue
• Forgot password
• Product availability
• Discounts
• Complaint
• Contact support

Type your question and click SEND.
"""
    )


# ==========================================
# BUTTON COMMANDS
# ==========================================

send_button.config(command=send_message)

clear_button.config(command=clear_chat)

save_button.config(command=save_chat)

help_button.config(command=help_window)


# ==========================================
# ENTER KEY SUPPORT
# ==========================================

user_input.bind("<Return>", send_message)
# ==========================================
# SET INITIAL FOCUS
# ==========================================

user_input.focus_set()

# ==========================================
# WINDOW CLOSE FUNCTION
# ==========================================

def on_closing():

    close = messagebox.askyesno(
        "Exit",
        "Do you really want to exit the chatbot?"
    )

    if close:
        root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

# ==========================================
# STATUS BAR
# ==========================================

status_bar = tk.Label(
    root,
    text="AI Customer Service Chatbot | Ready",
    bd=1,
    relief=tk.SUNKEN,
    anchor=tk.W,
    bg="#D6EAF8",
    fg="black",
    font=("Arial", 10)
)

status_bar.pack(side=tk.BOTTOM, fill=tk.X)

# ==========================================
# START APPLICATION
# ==========================================

root.mainloop()
