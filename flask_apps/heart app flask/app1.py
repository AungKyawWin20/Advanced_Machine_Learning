from flask import Flask,render_template,request
import numpy as np 
import pickle 
app=Flask(__name__)
@app.route("/")
def welcome():
    return "This is my first flask app."
# @app.route("/")
# def welcome():
#     return render_template('index.html')
if __name__=="__main__":
    app.run(debug=True)