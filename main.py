import json
import csv
import script
from adapterPattern import get_client
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime

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

categoriesVect = [cat.strip().lower().rstrip(".") for cat in script.categories]
categories = ",".join(categoriesVect)
realCategories = []
predictedCategories = []

client = get_client(provider_ai,model=modello_AI, generation_config=generation_config)

correctAnswer = 0 
tempo_risposte = []  
risultati = []
experiment = {}
validRowsNumber = 0
errorRowsNumber = 0

with open("dataset.csv", "r", newline="", encoding="utf-8") as fDataset:
    reader = csv.reader(fDataset)
    
    if next(reader, None) is not None:
        startTime = datetime.now()
        for i,fileLine in enumerate(reader,1):
            id,text,expected = fileLine

            prompt = ""

            if thereAreCategories:
                prompt = prompt_template.format(
                message=text, categories=categories)
            else:
                 prompt = prompt_template.format(
                    message=text)
            print("Linea corrente: " + str(i))
            try:
                response = client.generate(prompt)
            except Exception as e:
                print(f"[{i}] ERRORE: {e}", flush=True)
                errorRowsNumber+=1
                risultati.append({"input": text, "expected": expected.lower().strip(),
                          "predicted": "__error__", "correct": False,
                          "latency": "n/a", "error": str(e)})
                continue

            latency = response["latency"]

            tempo_risposte.append(latency)

            output = response["output"].strip().lower().rstrip(".")
            predictedCategories.append(output)
            realCategories.append(expected.strip().lower().rstrip("."))

            outputCorretto = output == expected.lower().strip().rstrip(".")

            if outputCorretto:   
                    correctAnswer+=1

            risultato = {
                "input": text,
                "expected": expected.lower().strip().rstrip("."),
                "predicted": output,
                "correct": outputCorretto,
                "latency": str(round(latency,2)) + "s"
            }

            risultati.append(risultato)
            validRowsNumber+=1

        endTime = datetime.now()

        durationBenchmark = endTime - startTime

        totalSeconds = durationBenchmark.total_seconds()

        minuts = totalSeconds // 60
        seconds = round(totalSeconds % 60,2)

        durationBenchmarkFormatted = f"{minuts} minuts and {seconds} seconds"

        tempo_medio_risposte = round(sum(tempo_risposte) / len(tempo_risposte), 2) if tempo_risposte else None

        accuracy = round((correctAnswer / validRowsNumber) * 100, 2) if validRowsNumber else None

        report = classification_report(realCategories, predictedCategories, labels=categoriesVect, zero_division=0, output_dict=True)
        print("--- Report ---")
        print(report)

        print("--- Matrice di Confusione ---")
        cm = confusion_matrix(realCategories, predictedCategories, labels=categoriesVect)
        print(cm)

        cm_structured = {}
        for i, label_true in enumerate(categoriesVect):
            cm_structured[label_true] = {categoriesVect[j]: int(cm[i][j]) for j in range(len(categoriesVect))}

        print("Risposte corrette: ", correctAnswer, "\nAccuracy: ", accuracy, "%")
        print(f"Tempo medio richieste: {tempo_medio_risposte}")

        righe_predette_fuori = sum(1 for p in predictedCategories if p not in categoriesVect)
        righe_in_matrice = sum(sum(row) for row in cm)
        righe_attese = validRowsNumber - righe_predette_fuori
        print(f"Matrice: {righe_in_matrice} righe, attese {righe_attese} "
            f"(validi={validRowsNumber}, predetti fuori={righe_predette_fuori})"
        )
        
        experiment = {
            "start time": startTime.strftime("%Y-%m-%d %H:%M:%S"),
            "end time": endTime.strftime("%Y-%m-%d %H:%M:%S"),
            "duration": durationBenchmarkFormatted,
            "ai provider":provider_ai,
            "model": modello_AI,
            "prompt": prompt_template,
            "accuracy": accuracy,
            "average_latency": f"{tempo_medio_risposte}s" if tempo_medio_risposte is not None else None,
            "valid rows":validRowsNumber,
            "error rows":errorRowsNumber,
            "predicted_rows_outside_labels": righe_predette_fuori,
            "results_note": (
            f"'results' contiene {validRowsNumber} risposte API valide + {errorRowsNumber} errori API. "
            f"'report' e 'matrix confusion' sono calcolati su {validRowsNumber - righe_predette_fuori} righe "
            f"(escluse {righe_predette_fuori} predizioni fuori dalle label note). "
            f"'accuracy' è calcolata su tutte le {validRowsNumber} risposte valide."
            ),
            "report": report,
            "matrix_rows_sum": righe_in_matrice,
            "matrix_rows_expected": righe_attese,
            "matrix_coherent": righe_in_matrice == righe_attese,
            "matrix confusion": cm_structured,
            "results": risultati
        }

        with open("risultati.json", "a", encoding="utf-8") as file:
            json.dump(experiment,file,ensure_ascii=False,indent=2)
    else:
        print("Il dataset è vuoto")