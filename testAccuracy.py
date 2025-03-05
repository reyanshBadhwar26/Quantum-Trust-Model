import openai
from dotenv import load_dotenv
import os 

#Initialize AI 
load_dotenv()
api_key = os.getenv("OPEN_AI_KEY")
openai.api_key = api_key

aiResponse = """

The Eiffel Tower is located in Paris, France, and was completed in 1889. It was designed by Gustave Eiffel. 
The Eiffel tower stands 330 meters tall, making it the tallest structure in the world. 
Albert Einstein invented the light bulb in 1879
The Great Wall of China is the only man-made structure visible from space.

"""

accuracyBackground=[{
    "role": "system", 
    "content": 
    """
    Your task is to analyze the text given to you and tell me its accuracy (i.e. fact check). Give it a score ranging from -1 to 1 depdening on how accurate it is.
    (e.g. Anything greater then 0 is going towards accuracy and anything below 0 is going towards inaccuracy). 
    Only give me a number and nothing else. 
    Fact check from a number of sources online to gather the score, and make sure to give a number that is as correct as possible.
    You can get a number in between 0, 1 and -1 as well if some parts of the text is true and others are false. 
    (For example, if you have text of 2 sentences and onoe of them is completely false while other is completely correct, it should be around 0.5)
    """
    }]

accuracyBackground.append({"role": "user", "content": aiResponse})
accuracyAnalysisAI= openai.chat.completions.create(model="gpt-4o-mini",messages=accuracyBackground)
accuracyAnalysis= accuracyAnalysisAI.choices[0].message.content
accuracyAnalysisVal= float(accuracyAnalysis)

print(f"Accuracy Analysis Score: {accuracyAnalysisVal}")