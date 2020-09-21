from flask import Flask # import flask
from flask import request
import nltk
from nltk.corpus import indian
from nltk.tag import tnt
import string
import re
from nltk.tag import tnt
from indicnlp.tokenize import indic_tokenize
import pickle
from flask import jsonify
app = Flask(__name__)             

@app.route("/")                   
def hello():                      
    return "Hello World!"         



@app.route("/getpos",methods=['POST'])
def getpos():
    #req_data = request.get_json()
    
    text = request.form['input_text']
    tokenized_text = indic_tokenize.trivial_tokenize(text)
    with open("./postagger.pkl",'rb') as f:
        pos_tagger = pickle.load(f)
    
    output = pos_tagger.tag(tokenized_text)
    op_dict = {}
    op_dict['output'] = output
    
    return jsonify(op_dict)
    


if __name__ == "__main__":        # on running python app.py
    app.run(debug=True)  