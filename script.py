import csv

fileInputs = "X_test.csv"
fileExpecteds = "y_test.csv"

categories=[]

def areListsCorrect(inputVect, expectedVect):
    idInputVect = [i["id"] for i in inputVect]
    idExpectedVect = [i["id"] for i in expectedVect]

    if len(inputVect) == len(expectedVect) and (len(idInputVect) == len(set(idInputVect)) and len(idExpectedVect) == len(set(idExpectedVect))):
        for input,expected in zip(inputVect,expectedVect):
            if input["id"] != expected["id"]:
                print("Dataset non corretto")
                return False
        return True
    else:
        print("Dataset non corretto")
        return False

def getOrderedDataFromFile(fileName):
    try:
        with open(fileName, "r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except FileNotFoundError:
        raise FileNotFoundError(f"File non trovato: {fileName}")
    except UnicodeDecodeError as e:
        raise ValueError(f"Encoding non valido in {fileName}: {e}")

    if not rows:
        raise ValueError(f"File vuoto o senza header: {fileName}")

    if "id" not in rows[0]:
        raise KeyError(
            f"Colonna 'id' assente in {fileName}. "
            f"Header trovato: {list(rows[0].keys())}. "
            f"Probabile delimitatore errato (atteso ',')"
        )

    try:
        return sorted(rows, key=lambda j: int(j["id"]))
    except (ValueError, KeyError) as e:
        raise ValueError(f"ID non numerico in {fileName}: {e}")

def writeDatasetFile(listInput,listExpected):
    with open("dataset.csv","w", newline="", encoding="utf-8") as fDatasest:
        writer = csv.writer(fDatasest)

        writer.writerow(["id","text","expected"])

        for input,expected in zip(listInput,listExpected):
            expected = expected["category_truth"]
            id = input["id"]
            input = input["text"]
            writer.writerow([id,input,expected])
            
            if not expected in categories:
                categories.append(expected)

def mergeDatasets():
    listInput = getOrderedDataFromFile(fileInputs)
    listExpected = getOrderedDataFromFile(fileExpecteds)

    if (listInput is not None and listExpected is not None) and areListsCorrect(listInput,listExpected):
        writeDatasetFile(listInput,listExpected)
    else:
        print("Errore di lettura dei file dataset")

mergeDatasets()