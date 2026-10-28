import os
from dotenv import load_dotenv
from google import genai
import time
import json
import csv
import script

modello_AI = "gemini-flash-lite-latest"
thinking_level = "medium"

prompt_template = """You are an expert in the analysis and classification of IT tickets.

Analyze the following ticket and identify the category that most accurately describes the main issue reported.

You are provided with a predefined list of categories; you must choose the category beetween the list of categories provided.

Rules:
- Choose exactly one category from the provided categories list.
- Do not create new categories or modify their names.
- Return only the selected category name.

Ticket:
{message}
Categories:
{categories}"""

categories = ",".join(script.categories)

print(categories)

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
)

generation_config = {
    'max_output_tokens': 65536,
    'thinking_level': thinking_level,
}

correctAnswer = 0 
tempo_risposte = []  
risultati = []
experiment = {}
rowsNumber = 0

with open("dataset.csv", "r", newline="", encoding="utf-8") as fDataset:
    reader = csv.reader(fDataset)
    
    if next(reader, None) is not None:
        for fileLine in reader:
            rowsNumber+=1

            if rowsNumber == 10:
                break

            id,input,expected = fileLine
            print(fileLine)
            start = time.perf_counter()

            prompt = prompt_template.format(
            message=input, categories=categories)

            interaction = client.interactions.create(
                model=f'models/{modello_AI}',
                input=prompt,
                generation_config=generation_config,
            )

            elapsed = time.perf_counter() - start

            tempo_risposte.append(elapsed)

            output = interaction.output_text

            output = output.strip().lower().rstrip(".")

            outputCorretto = False

            outputCorretto = output == expected.lower()

            if outputCorretto:   
                    correctAnswer+=1

            risultato = {
                "input": input,
                "expected": expected,
                "predicted": output,
                "correct": outputCorretto,
                "latency": round(elapsed,2)
            }

            risultati.append(risultato)

            print(risultato)

        tempo_medio_risposte = sum(tempo_risposte)/len(tempo_risposte)
        accuracy = (correctAnswer/rowsNumber) * 100
        precision = 0
        recall = 0
        f1 = 0

        print("Risposte corrette: ", correctAnswer, "\nAccuracy: ", accuracy, "%")
        print(f"Tempo medio richieste: {tempo_medio_risposte:.3f}")

        experiment = {
            "model": f"models/{modello_AI}",
            "prompt": prompt_template,
            "thinking_level":"medium",
            "accuracy": accuracy,
            "average_latency": tempo_medio_risposte,
            "results": risultati
        }

        with open("risultati.json", "w", encoding="utf-8") as file:
            json.dump(experiment,file,ensure_ascii=False,indent=2)
    else:
        print("Il dataset è vuoto")