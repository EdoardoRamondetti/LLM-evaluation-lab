import csv

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
    file = open(fileName, "r", newline="", encoding="utf-8")
    try:
        listDataFile = sorted(list(csv.DictReader(file)), key=lambda j: int(j["id"]))
        return listDataFile
    except ValueError as e:
        print("Errore conversione id in numero")
    finally:
        file.close()

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
    listInput = getOrderedDataFromFile("X_test.csv")
    listExpected = getOrderedDataFromFile("y_test.csv")

    if (listInput is not None and listExpected is not None) and areListsCorrect(listInput,listExpected):
        writeDatasetFile(listInput,listExpected)

mergeDatasets()