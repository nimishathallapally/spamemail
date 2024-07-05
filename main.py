import string

import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

nltk.download('stopwords')

#Reading the dataset and initialising

data_set=pd.read_csv('spam_ham_dataset.csv')

#replace the new line with whitespaces using the lambda function
data_set['text']=data_set['text'].apply(lambda x: x.replace('\r\n',' '))

stemmer=PorterStemmer()
corpus=[]

stopword_set=set(stopwords.words('english'))

#Preprocessing and Stemming

for i in range(len(data_set)):

    #change everything to lowercase
    text=data_set['text'].iloc[i].lower()

    #remove punctuation
    text=text.translate(str.maketrans('','',string.punctuation)).split()

    #stem only if it not part of stopwords
    for word in text:
        if word not in stopword_set:
            text=stemmer.stem(word)

    text=''.join(text)
    corpus.append(text)

# vectorize all of this

vectorizer=CountVectorizer()

x=vectorizer.fit_transform(corpus).toarray()

# This x vectorises corpus and is turned into an array so you have a array of numbers stored in x

y=data_set.label_num

# train_test_split(x, y, test_size=0.2) randomly splits x and y 
# such that 80% of the data is allocated to x_train and y_train, and 20% to x_test and y_test.

# x_train: This is the subset of x used for training the model.
# x_test: This is the subset of x used for testing the model's performance.
# y_train: This is the subset of y corresponding to x_train, used to train the model.
# y_test: This is the subset of y corresponding to x_test, used to evaluate the model's performance.

x_train, x_test, y_train, y_test = train_test_split(x ,y ,test_size=0.2)

# n_jobs=-1 is a shorthand that tells scikit-learn to use all available cores.
clf=RandomForestClassifier(n_jobs=-1)

# clf.fit(x_train, y_train) trains the RandomForestClassifier model clf using the training data (x_train and y_train), 
# enabling it to learn from the provided features and labels.
clf.fit(x_train , y_train)

accuracy=clf.score(x_test,y_test)

print(accuracy)