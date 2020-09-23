# ३९ गेंदों में दो चौकों और एक छक्के की मदद से ३४ रन बनाने वाले परोरे अंत तक आउट नहीं हुए।
from flask import Flask
from flask import request
import nltk
from nltk.corpus import indian
from nltk.tag import tnt
import string
import re
from nltk.tag import tnt
from indicnlp.tokenize import indic_tokenize
import pickle
from flask import jsonify, render_template
app = Flask(__name__)      

@app.route("/")                   
def hello():                      
    return render_template('index.html')

@app.route("/getpos",methods=['POST'])
def getpos():
    text = request.form['input_text']
    tokenized_text = indic_tokenize.trivial_tokenize(text)
    with open("./postagger.pkl",'rb') as f:
        pos_tagger = pickle.load(f)
    output = pos_tagger.tag(tokenized_text)
    return render_template('result.html',data = output)

if __name__ == "__main__":
    app.run(debug=True)

'''
CC coordinating conjunction
CD cardinal digit
DT determiner
EX existential there (like: "there is" ... think of it like "there exists")
FW foreign word
IN preposition/subordinating conjunction
JJ adjective 'big'
JJR adjective, comparative 'bigger'
JJS adjective, superlative 'biggest'
LS list marker 1)
MD modal could, will
NN noun, singular 'desk'
NNS noun plural 'desks'
NNP proper noun, singular 'Harrison'
NNPS proper noun, plural 'Americans'
PDT predeterminer 'all the kids'
POS possessive ending parent's
PRP personal pronoun I, he, she
PRP$ possessive pronoun my, his, hers
RB adverb very, silently,
RBR adverb, comparative better
RBS adverb, superlative best
RP particle give up
TO to go 'to' the store.
UH interjection errrrrrrrm
VB verb, base form take
VBD verb, past tense took
VBG verb, gerund/present participle taking
VBN verb, past participle taken
VBP verb, sing. present, non-3d take
VBZ verb, 3rd person sing. present takes
WDT wh-determiner which
WP wh-pronoun who, what
WP$ possessive wh-pronoun whose
WRB wh-abverb where, when
'''