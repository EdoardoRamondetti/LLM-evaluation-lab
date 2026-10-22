import os
from dotenv import load_dotenv
from google import genai
import time

load_dotenv()

dataset = [
    {
        "input": "Non riesco ad accedere al mio account",
        "expected": "account"
    },
    {
        "input": "La stampante non stampa",
        "expected": "printer"
    },
    {
        "input": "Ho dimenticato la password",
        "expected": "account"
    },
    {
        "input": "Il computer non si accende",
        "expected": "hardware"
    }
]

predictions = [
        "account",
        "printer",
        "password",
        "hardware"
    ]

def evaluate(dataset, predictions):
    if len(dataset) == len(predictions):
        correctCase = 0

        if len(dataset) != 0:
            for case, prediction in zip(dataset, predictions):
                expected = case["expected"]
                if expected.lower() == prediction.lower():
                    correctCase+=1
            accuracy = correctCase/len(dataset)
        
            return {"accuracy": accuracy}
        else:
            raise ValueError("Dataset is empty")
    else:
        ValueError("Dataset and predictions don't have the same length")



client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
)

generation_config = {
    'max_output_tokens': 65536,
    'thinking_level': 'medium',
}

correctAnswer = 0

for data in dataset:
    interaction = client.interactions.create(
        model='models/gemini-flash-lite-latest',
        input=f"""Rispondi con una sola parola: quale categoria assegneresti a questo messaggio? 
        {data["input"]}. Scegli tra {",".join(predictions)}.""",
        generation_config=generation_config,
    )

    output = interaction.output_text

    output = output.strip().lower().rstrip(".")

    if output == data["expected"]:
        correctAnswer+=1

print("Risposte corrette: ", correctAnswer, "\nAccuracy: ", (correctAnswer/len(dataset))*100, "%")