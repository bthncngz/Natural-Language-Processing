# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 16:57:44 2023

@author: Batuhan
"""

import numpy as np
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter("ignore")
data = pd.read_csv("C:\\Users\\Batuhan\\Desktop\\Language_Detection.csv")
data.head(10)
data["Language"].value_counts()

X = data["Text"]
y = data["Language"]

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

data_list = []

for text in X:
    
    text = re.sub(r'[!@#$(),\n"%^*?\:;~`0-9]', ' ', text)
    text = re.sub(r'[[]]', ' ', text)
    
    text = text.lower()
    
    data_list.append(text)
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(data_list).toarray()
                     
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

def predict(text):
    x = cv.transform([text]).toarray()
    lang = model.predict(x)
    lang = le.inverse_transform(lang)
    print("Bu dil",lang[0])
    
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

ac = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)

print("Accuracy is :",ac)
print(cr)

plt.figure(figsize=(15,10))
sns.heatmap(cm, annot = True)
plt.show()

    
                                           