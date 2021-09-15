from flask import Flask, request
from pymongo import MongoClient

app = Flask(__name__)

conn = MongoClient(host="db", port=27018)
connObj = conn["testdb1"]["testColl"]


@app.route("/", methods=['POST'])
def loadData():
    data = request.json
    print(data)
    connObj.insert_one(data)
    return "Success" + str(data)


app.run(host="0.0.0.0", port=5001)
