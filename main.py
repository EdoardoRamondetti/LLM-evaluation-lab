import os
from dotenv import load_dotenv
from google import genai
import time
import json

modello_AI = "gemini-flash-lite-latest"
thinking_level = "medium"

categories = []
data = {"input":""}

prompt_template = """Rispondi con una sola parola: quale categoria assegneresti a questo messaggio?
{messaggio}
Scegli tra {categorie}."""

prompt = f"""Rispondi con una sola parola: quale categoria assegneresti a questo messaggio? 
            {data["input"]}. Scegli tra {",".join(categories)}."""

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
        "expected": "password"
    },
    {
        "input": "Il computer non si accende",
        "expected": "hardware"
    }
]

categories = [
        "account",
        "printer",
        "password",
        "hardware"
    ]

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

if len(dataset) > 0:
    for data in dataset:
        start = time.perf_counter()

        prompt = prompt_template.format(
        messaggio=data["input"],
        categorie=", ".join(categories)
        )
        print(prompt)

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

        outputCorretto = output == data["expected"]

        if outputCorretto:   
                correctAnswer+=1

        risultato = {
            "input": data["input"],
            "expected": data["expected"],
            "predicted": output,
            "correct": outputCorretto,
            "latency": round(elapsed,2)
        }

        risultati.append(risultato)

    tempo_medio_risposte = sum(tempo_risposte)/len(tempo_risposte)
    accuracy = (correctAnswer/len(dataset)) * 100

    print("Risposte corrette: ", correctAnswer, "\nAccuracy: ", accuracy, "%")
    print(f"Tempo medio richieste: {tempo_medio_risposte:.3f}")
    print(f"Risulati:  {risultati}")

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