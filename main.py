import json
import csv
import script
from adapterPattern import get_client

provider_ai = "gemini"
modello_AI = "gemini-flash-lite-latest"
max_tokens = 300
temperature = 0.0
# True se si vuole dare le categorie tra cui scegliere la risposta al AI
thereAreCategories = True

generation_config = {
    'max_tokens': max_tokens,
    "temperature":temperature
}

promptTextCategories = "\nCategories:\n{categories}" if thereAreCategories else ""

prompt_template = """You are an expert in the analysis and classification of IT tickets.

Analyze the following ticket and identify the category that most accurately describes the main issue reported.

You are provided with a predefined list of categories; you must choose the category beetween the list of categories provided.
If there is not a list of categories you have to obtain the category from the ticket

Rules:
- Choose exactly one category from the provided categories list.
- Do not create new categories or modify their names.
- Return only the selected category name do not write anything else only the category name and nothing else.

Ticket:
{message}""" + promptTextCategories

categories = ",".join(script.categories)

print(categories)

client = get_client(provider_ai,model=modello_AI, generation_config=generation_config)

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

            if rowsNumber == 3:
                break

            id,input,expected = fileLine
            print(fileLine)

            prompt = ""

            if thereAreCategories:
                prompt = prompt_template.format(
                message=input, categories=categories)
                print(prompt)
            else:
                 prompt = prompt_template.format(
                    message=input)

            print(prompt)

            response = client.generate(prompt)

            latency = response["latency"]

            tempo_risposte.append(latency)

            output = response["output"].strip().lower().rstrip(".")

            print(output)

            outputCorretto = output == expected.lower()

            if outputCorretto:   
                    correctAnswer+=1

            risultato = {
                "input": input,
                "expected": expected.lower(),
                "predicted": output,
                "correct": outputCorretto,
                "latency": str(round(latency,2)) + "s"
            }

            risultati.append(risultato)

            print(risultato)

        tempo_medio_risposte = round(sum(tempo_risposte)/len(tempo_risposte),2)
        accuracy = round((correctAnswer/rowsNumber) * 100,2)
        precision = 0
        recall = 0
        f1 = 0

        print("Risposte corrette: ", correctAnswer, "\nAccuracy: ", accuracy, "%")
        print(f"Tempo medio richieste: {tempo_medio_risposte}")

        experiment = {
            "model": modello_AI,
            "prompt": prompt_template,
            "accuracy": accuracy,
            "average_latency": str(tempo_medio_risposte) + "s",
            "results": risultati
        }

        with open("risultati.json", "w", encoding="utf-8") as file:
            json.dump(experiment,file,ensure_ascii=False,indent=2)
    else:
        print("Il dataset è vuoto")