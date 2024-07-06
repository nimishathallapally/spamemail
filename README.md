# Spam Email Detection Project

## Overview
This project is a spam email detection system using machine learning techniques. The model here is then tested with the classifier RandomForestClassifier. The dataset used to train and test the model comes from Kaggle. Additionally, a frontend interface is implemented using Flask, HTML, and CSS to classify user-provided emails as spam or not spam.

## Dataset
Kaggle dataset named spam_ham_dataset.csv, which contains the labeled emails classified into spam and ham. Each email is represented in the dataset along with its text and the corresponding label.

## Libraries Used
- `numpy` and `pandas` for data manipulation and analysis.
- `nltk` (Natural Language Toolkit) for performing natural language processing tasks.
- `sklearn` (scikit-learn) for applying machine learning algorithms and other tools.
- `Flask` for building the web application.

## Steps Preprocessing
1. **Read the Dataset**: The dataset (`spam_ham_dataset.csv`) was loaded using the pandas library.
2. **Text Cleaning**:
   - Convert text to lowercase.
   - Remove punctuation using Python's `string.punctuation`.
   - Replace newline characters (`\n`) with whitespaces.
3. **Stemming**: Reducing words to their root form using Porter Stemmer from `nltk`.
4. **Stopwords Removal**: Removing common English stopwords using `nltk`'s stopwords corpus.
5. **Vectorization**: Converting text data into numerical feature vectors using `CountVectorizer` from `sklearn`.

## Training the Model
- **Splitting Data**: Using `train_test_split` from `sklearn` to split the dataset into 80% training and 20% testing.
- **Model Selection**: Apply a `RandomForestClassifier` with `n_jobs=-1` for parallelizing over all CPU cores in order to speed up the process.
- **Model Training**: Fit the `RandomForestClassifier` on the training data (`x_train`, `y_train`).
- **Model Evaluation**: Test the model using `.score()` on the test data.


## Frontend Implementation
- **Flask Web Application**: Adding a frontend interface using Flask to create a web-based email classification tool.
- **HTML and CSS**: Designing the user interface HTML and CSS.
- **Display Results**: Showing the classification result (spam or not spam) to the user on the frontend interface.

## Integration with joblib
- **Loading Models**: Using `joblib` to load pre-trained models (`vectorizer.pkl` and `classifier.pkl`) within the Flask application for efficient classification of user inputs.

## Results
- The accuracy of the trained model for test data is printed and further analyzed for performance improvement.
- User-friendly interface allows users to input email text and receive classification results.

## Usage
- Run below command to ensure all dependency packages are installed: `pip install numpy pandas nltk scikit-learn joblib flask`
- Ensure the dataset (`spam_ham_dataset.csv`), trained models (`vectorizer.pkl` and `classifier.pkl`), and Flask application (`app.py`, `templates/`, `static/`) are correctly set up and accessible.
- Run the Flask application (`app.py`) to start the web server.

## Dependencies
- Python 3.x
- numpy
- pandas
- nltk
- scikit-learn
- joblib
- flask

## Acknowledgements
- Dataset sourced from Kaggle: [Spam/ham email classification dataset](https://www.kaggle.com/datasets/venky73/spam-mails-dataset/data).
