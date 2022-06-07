# -*- coding: utf-8 -*-
import pandas as pd
from flask import Flask, render_template, jsonify, request, redirect
#from custom_classes import PreprocessingDataTransformer, DensifierTransformer
import joblib
import __main__
#custom_classes.py
from sklearn.base import BaseEstimator, TransformerMixin
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk


app = Flask(__name__)
global classificator, tags_dictionnaires

@app.route('/', methods=['GET','POST'])
def predict():

    if request.method=="POST":

         req=request.form

         #J'extrais de la requête les données
         titre = req.get("titre")
         question = req["question"]

         #Je rassemble le titre et la question dans un même objet string
         raw_text = titre + " " + question

         #Import du classificateur entrainé
         classificator=get_clf()

         #Je prédis les tags associés à la question
         prediction = classificator.predict([[raw_text]])
         prediction = prediction[0]

         #Je convertis les prédictions en tags
         tag_label = ""
         tags = []

         if sum(prediction)==0:
            prediction[-1]=1

         #import de la liste qui permet de convertie les ID_tags prédits en tags
         tags_dictionnaires=get_tags_dictionnaires()

         #J'initialise un compteur
         i=0
         for tag_id in prediction:
            if tag_id!=0:
                tag_label=tags_dictionnaires[i]
                tags.append(tag_label)

            i+=1

         print(tags)
         return render_template("question_form.html",titre=titre,question=question,tags=', '.join(tags))

    return render_template("question_form.html")


def get_clf():
    #if classificator is None:
    classificator=joblib.load("tagging_classifieur.joblib")

    return classificator

def get_tags_dictionnaires():
    #if tags_dictionnaires is None:
    tags_dictionnaires=joblib.load("liste_tags_retenus.joblib")

    return tags_dictionnaires


class PreprocessingDataTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # X_df = pd.DataFrame(X, columns=['Body'])

        # CleanedDataset = pd.DataFrame(X, columns=['Body']).apply(body_to_words)
        CleanedData = []

        for body in X:
            new_text = body_to_words(body[0])
            CleanedData.append(new_text)

        return CleanedData

class DensifierTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.toarray()
        # return self

def body_to_words(raw_text):
    # 1. Retirer les balises HTML
    review_text = BeautifulSoup(raw_text,"html.parser").get_text()
    #
    # 2. Retirer les caractères non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    #
    # 3. convertir les lettres en minuscules, split into individual words
    words = letters_only.lower().split()


    # 4. Création d'un set de stopwords
    stops = set(stopwords.words("english"))

    # 5. Retirer les stopwords du body
    words = [w for w in words if not w in stops]

    # 6. Retirer les répétitions de lettres type : "aaa", "bbbbbbb", etc...
    words = [w for w in words if bool(re.search(r"(.)\1+",w))==False]


    #J'initialise un lemminizer celui-ci va plus loin qu'un stemmer en effectuant un contextualisation
    lemmatizer = WordNetLemmatizer()

    # loop for stemming each word
    # in string array at ith row
    meaningful_words = [lemmatizer.lemmatize(word) for word in words]

    # 6. Jointure de tous les mots conservés et retour du résultat
    return( " ".join(meaningful_words))

__main__.PreprocessingDataTransformer = PreprocessingDataTransformer
__main__.DensifierTransformer = DensifierTransformer

if __name__ == '__main__':
    print("__name__ is __main__")
    app.run(debug=True)

