# Quantum Trust Model

The **Quantum Trust Model** is an interactive research prototype that explores how **trust in human–AI interactions evolves over time** using concepts inspired by **quantum probability and state evolution**.

Rather than treating trust as a static or purely linear value, this project models trust as a **quantum state**, allowing it to evolve dynamically based on:
- a user’s initial trust
- risk associated with the situation
- prior knowledge
- ongoing dialogue between a human and an AI system

The system visualizes how trust changes in real time during a Human-AI interaction.

---

## Key Features

- Quantum-inspired trust modeling using single-qubit state representations  
- Real-time trust evolution visualization  
- Interactive Streamlit web interface  
- Live AI interaction using OpenAI’s GPT models  
- Designed as a research and exploratory tool  

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Quantum-Trust-Model.git
cd Quantum-Trust-Model
```

### 2. Create and activate a virtual environment (recommended)

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Your OpenAI API Key

Create a `.env` file in the project root and add your API key:

```bash
OPEN_AI_KEY=your_openai_api_key_here
```

## Running the Application

Launch the Streamlit app:
```bash
streamlit run app.py
```
Open the local URL shown in your terminal.

--- 

## How to Use the App

1. Initialize trust parameters using the sliders:
   - **Initial Trust**
   - **Situation Risk**
   - **Prior Knowledge**

2. Click **Confirm** to initialize the quantum trust state.

3. Begin chatting with the AI.

4. Observe:
   - AI responses
   - Real-time updates to the trust evolution graph

5. Use the sidebar to:
   - Mark a task as completed
   - Start a new chat session
