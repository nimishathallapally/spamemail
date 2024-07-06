from flask import Flask,request,render_template
import joblib
import string

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load the vectorizer and Classifier
vectorizer=joblib.load('vectorizer.pkl')
clf=joblib.load('classifier.pkl')

# Preprocessing Function
def preprocessing_email(text):
    stemmer=PorterStemmer()
    stopword_set=set(stopwords.words('english'))

    # make everything lowercase
    text=text.lower()

    #remove the punctuation
    text=text.translate(str.maketrans('','',string.punctuation)).split()

    #stem only if it not part of stopwords
    for word in text:
        if word not in stopword_set:
            text=stemmer.stem(word)

    return ''.join(text)

# Function to classify the email

def classify_email(email,vectorizer,clf):
    processed_email=preprocessing_email(email)

    # preprocessed email is converted into numerical vector using the previously trained vectorizer
    email_vector=vectorizer.transform([processed_email]).toarray()

    # .predict() method uses the trained model to predict the class label for the given vectorized email.
    prediction=clf.predict(email_vector)

    # the prediction is an array with a single element, prediction[0] extracts this element and returns it as the output of the function.
    return prediction[0]

# Initialise Flask App
app=Flask(__name__)

@app.route('/')

def home():
    return render_template('index.html')

@app.route('/classify',methods=['POST'])
def classify():
    email=request.form['email']
    prediction_result=classify_email(email,vectorizer,clf)
    if prediction_result == 1:
        result="SPAM"
    else:
        result="NOT SPAM"
    
    return render_template('result.html',email=email,result=result)


if __name__ == '__main__':
    app.run(debug=True)