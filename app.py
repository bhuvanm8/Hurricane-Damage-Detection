from flask import Flask,render_template,redirect,request
import numpy as np
from PIL import Image
from tensorflow import expand_dims
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "/Users/bhuvanm/Desktop/flooding/static"

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict",methods=["GET","POST"])
def predictPage():
    if request.method == "POST":
        img = request.files["im"]
        fName = img.filename.split(".")
        fName = "myimg" + "." + fName[-1]
        img.save(os.path.join(app.config["UPLOAD_FOLDER"],fName))
        pic = Image.open(img)
        pic = np.array(pic.resize((256,256)))
        pic = expand_dims(pic,axis=0)
        model = load_model("flood.h5")
        pred = model.predict(pic)
        pred_sent = "Damaged" if pred[0][0] <= 0.5 else "Not damaged"
        return render_template("predict.html",pred_sent=pred_sent,imName=fName)
    return render_template("predict.html")

if __name__=="__main__":
    app.run(debug=True)