import streamlit as st
import openai
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import trustModel as trust
import time  # Import time module

# Initialization
load_dotenv()
api_key = os.getenv("OPEN_AI_KEY")
openai.api_key = api_key

# Streamlit App
st.set_page_config(page_title="Quantum-Enhanced AI Chatbot", layout="wide")
st.title("Quantum-Enhanced AI Chatbot")
st.write("Interact with the AI and watch the trust levels evolve in real-time!")

# Persistent State for Trust Scores and Timer
if "trustScoresRatio" not in st.session_state:
    st.session_state.trustScoresRatio = []

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": """
                                  You are a kind and helpful assistant. Prompts from the user will usually include something related to an emergency situation (for example bleeding, choking, heart attack etc.)
                                  You should always try to give the quickest and best response possible to get them out of the situation and tell them steps on how they could solve their problem."""}]

if "button_pressed" not in st.session_state:
    st.session_state.button_pressed = False

if "start_time" not in st.session_state:
    st.session_state.start_time = None  # Initialize start time

if not st.session_state.button_pressed:

    # User Input for Initial Trust
    initialTrust = st.slider("From 0-10, how much do you trust this AI system?", 0, 10, 5)

    situationRisk = st.slider("From 0-10, what is the risk involved in the situation?", 0, 10, 5)

    priorKnowledge = st.slider("From 0-10, how much prior knowledge do you have about the topic?", 0, 10, 5)

    if st.button("Confirm"):
        st.session_state.button_pressed = True  
        st.session_state.start_time = time.time()  # Start the timer

        trust.initialTrust(initialTrust)
        counts = trust.getSingleQubitProbabilities()
        ratio = counts * 100
        st.session_state.trustScoresRatio.append(ratio)

        st.rerun()

else:

    # Chat Input and Output
    user_input = st.text_input("Your Message:")

    if st.button("Send"):
        if user_input:
            # Add User Message
            st.session_state.messages.append({"role": "user", "content": user_input})

            # Get AI Response
            chat_response = openai.chat.completions.create(model="gpt-4o-mini",messages=st.session_state.messages)
            reply = chat_response.choices[0].message.content
            st.write(f"AI: {reply}")
            st.session_state.messages.append({"role": "assistant", "content": reply})

            trust.sentimentAnalysis(user_input)
            counts = trust.getSingleQubitProbabilities()
            ratio = counts * 100
            st.session_state.trustScoresRatio.append(ratio)

    # Plot Trust Scores in Real-Time
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(len(st.session_state.trustScoresRatio)), st.session_state.trustScoresRatio, label="Trust Score Ratio")
    ax.set_xticks(range(len(st.session_state.trustScoresRatio)))
    ax.set_yticks(list(range(0, 101, 10)))
    ax.set_title("Trust Level Evolution")
    ax.set_xlabel("Interaction Count")
    ax.set_ylabel("Trust Score")
    ax.legend()
    buf = BytesIO()
    plt.savefig(buf, format="png")
    st.image(buf)

    # Timer Display in the Sidebar
    elapsed_time = "--:--"
    if st.session_state.start_time:
        total_elapsed = time.time() - st.session_state.start_time
        minutes, seconds = divmod(int(total_elapsed), 60)
        elapsed_time = f"{minutes:02d}:{seconds:02d}"

    # Sidebar Layout
    with st.sidebar:
        st.header("Options")
        st.write(f"Elapsed Time: {elapsed_time}")
        if st.button("Task Completed"):
            st.success("Task marked as completed!")
        if st.button("New Chat"):
            st.session_state.button_pressed = False  
            st.rerun()