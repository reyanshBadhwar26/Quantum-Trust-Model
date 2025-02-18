import os
import openai
from dotenv import load_dotenv
import math
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_bloch_multivector
import matplotlib.pyplot as plt
from qiskit.quantum_info import partial_trace, DensityMatrix
import numpy as np

#Initialize AI 
load_dotenv()
api_key = os.getenv("OPEN_AI_KEY")
openai.api_key = api_key

#Initialize Classic Simulator -Not used in this code
simulator = AerSimulator()

#Initialize Variables
totalAngle = 0
counter = 1
totalSentimentAngle = 0

# Quantum Circuit Initialization - 0 for trust, 1 for emotion, 2 for reliance, 3 for task completion, 4 for clarity, 5 for accuracy
qBits = QuantumCircuit(6)

#Apply Superposition
qBits.h(0)

def showCircuit():
    print(qBits.draw())

def initialTrust(userInput):

    theta = (((userInput-5)*0.2)*math.pi/2)

    if userInput == 0 or userInput == 10:
        theta = (((userInput-5)*0.19)*math.pi/2)

    qBits.ry(theta, 0)

def riskAnalysis(userInput):

    theta = 0

    qBits.ry(theta, 0)


def priorKnowledgeAnalysis(userInput):

    theta = 0

    qBits.ry(theta, 2)

def sentimentAnalysis(userInput):
    global totalAngle, counter, totalSentimentAngle

    totalAngle = getCurrentAngle(qBits, 0)

    print(f"Total sentiment angle is: {totalSentimentAngle}")
    print(f"Total angle is: {totalAngle}")
    
    sentimentBackground=[{
        "role": "system", 
        "content": 
        """
        Your task is to analyze the sentence given to you and tell me its sentiment. Give it a score ranging from -1 to 1 depdening on how positive it is.
        (e.g. Anything greater then 0 is positve and anything below 0 is negative AND 0 is exactly neutral). 
        Only give me a number and nothing else. For example if I put I hate you that is very negative so the answer would be very close to -1.
        The closer to the 1 or -1 the better the sentiment.
        Factor in the exclamation marks, extra question marks for frustation, capitalizations for urgency and frustation etc. All of these could affect the sentiment. 
        If the user is just asking a simple question (for example, I am choking, what should I do?) that is near to neutral (0).
        IT IS IMPORTANT THAT YOU FACTOR IN THE CONTEXT WHILE DECIDING ON A SENTIMENT. MAKE SURE TO LOOK AT THE CONVERSATIONS ABOVE THE ONE BEING JUDGED IN ORDER TO GIVE A NUMBER.
        """
        }]
    
    sentimentBackground.append({"role": "user", "content": userInput})
    sentimentAnalysisAI= openai.chat.completions.create(model="gpt-4o-mini",messages=sentimentBackground)
    sentimentAnalysis= sentimentAnalysisAI.choices[0].message.content
    sentimentAnalysisVal= float(sentimentAnalysis)
    print(sentimentAnalysisVal)

    # Calculate the remaining angle for sentiment adjustment
    leftOverTheta = math.pi - totalSentimentAngle 

    if totalSentimentAngle < 0:
        leftOverTheta = -(-math.pi - totalSentimentAngle)
    
    theta = sentimentAnalysisVal * leftOverTheta /counter

    print(f"SentimentTheta: {theta}")

    qBits.ry(theta, 1)

    totalSentimentAngle += theta

    if sentimentAnalysisVal < 0:
        qBits.cry((0 - totalAngle)/counter, 1, 0)
    elif sentimentAnalysisVal > 0:
        qBits.cry((math.pi - totalAngle)/counter, 1, 0)

    showCircuit()
    
    totalAngle = getCurrentAngle(qBits, 0)

    print(f"Updated total trust angle: {totalAngle}")
    counter += 0.5

def updateTaskCompletion():
    pass

def updateClarity():
    pass

def updateAccuracy():
    pass

def updateReliance():
    #reliance is based on task completion rate, accuracy and clarity of response
    pass

#Will need to organize the code, so only one function needs to be called from the front end - Only when we know all backend functions are working
def updateTrust():
    pass


def getSingleQubitProbabilities():
    # Obtain the statevector from the quantum circuit
    state = Statevector.from_instruction(qBits)
    
    # Compute the reduced density matrix for qubit 0 by tracing out qubit 1
    reduced_dm = partial_trace(state, [1])
    
    # Convert the reduced state to a DensityMatrix
    dm = DensityMatrix(reduced_dm)
    
    # Extract the diagonal elements to get probabilities
    prob_0 = np.real(dm.data[0, 0])  # Element corresponding to |0><0|
    prob_1 = np.real(dm.data[1, 1])  # Element corresponding to |1><1|
    
    print(f"Probability of qubit 0 being 0: {prob_0}")
    print(f"Probability of qubit 0 being 1: {prob_1}")
    
    return prob_1

def showQubit():
    
    state = Statevector.from_instruction(qBits)

    # Step 5: Visualize the state on the Bloch sphere
    plot_bloch_multivector(state.data)

    # Step 6: Display the plot
    plt.show()

def getCurrentAngle(qBits, qubit_index=0):
    # Obtain the state vector from the quantum circuit
    state = Statevector.from_instruction(qBits)
    
    # Compute the reduced density matrix for the qubit of interest
    reduced_dm = partial_trace(state, [i for i in range(qBits.num_qubits) if i != qubit_index])
    
    # Convert to a DensityMatrix object
    dm = DensityMatrix(reduced_dm)
    
    # Access the elements of the density matrix
    rho = dm.data  # Get the NumPy array representing the density matrix
    
    # Extract real and imaginary parts of the elements to compute Bloch vector
    rho00 = np.real(rho[0, 0])
    rho01 = rho[0, 1]
    rho10 = rho[1, 0]
    rho11 = np.real(rho[1, 1])
    
    # Compute Bloch vector components
    x = 2 * np.real(rho01)
    y = 2 * np.imag(rho01)
    z = rho00 - rho11
    
    # Calculate the angle theta from the Z-axis projection
    theta = np.arccos(z)  # z is the Z component of the Bloch vector
    
    # Return the angle in radians
    return theta