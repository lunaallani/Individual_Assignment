from flask import Flask, request
from bikes.model import train_and_persist, predict

app = Flask(_name_)

@app.route("/")
def home():
    return {
        "message": "Hello world! Hey",
        "version": "0.1"  
           }

@app.route("/train_and_persist")
def train_and_persist():
    print(request.args)
    sentence = request.args["sentence"]
    lower = request.args.get("lower", False)
    return str(tokenize(sentence, lower = lower))

if _name_ == "_main_":
    import os 
    port = int(os.environ["PORT"])
    app.run(host = "0.0.0.0", port=port)