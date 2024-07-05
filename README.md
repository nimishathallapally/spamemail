# Spam Email Detection Project

## Overview
This project aims to develop a machine learning model for detecting spam emails using a RandomForestClassifier. The dataset used for training and testing is obtained from Kaggle.

## Dataset
The dataset (`spam_ham_dataset.csv`) used in this project is sourced from Kaggle, containing labeled emails classified as spam or ham (non-spam). Each email is represented by its text content and a corresponding label.

## Libraries Used
- `numpy` and `pandas` for data manipulation and analysis.
- `nltk` (Natural Language Toolkit) for natural language processing tasks.
- `sklearn` (scikit-learn) for machine learning models and tools.
  
## Preprocessing Steps
1. **Reading the Dataset**: Loading the dataset (`spam_ham_dataset.csv`) using pandas.
2. **Text Cleaning**:
   - Converting text to lowercase.
   - Removing punctuation using Python's `string.punctuation`.
   - Replacing newline characters (`\r\n`) with whitespaces.
3. **Stemming**: Reducing words to their root form using Porter Stemmer from `nltk`.
4. **Stopwords Removal**: Filtering out common English stopwords using `nltk`'s stopwords corpus.
5. **Vectorization**: Converting text data into numerical feature vectors using `CountVectorizer` from `sklearn`.

## Training the Model
- **Splitting Data**: Using `train_test_split` from `sklearn` to divide the dataset into training and testing sets (80% training, 20% testing).
- **Model Selection**: Implementing a RandomForestClassifier with `n_jobs=-1` to utilize all available CPU cores for faster training.
- **Model Training**: Fitting the RandomForestClassifier on the training data (`x_train` and `y_train`).
- **Model Evaluation**: Evaluating the model accuracy on the test data (`x_test` and `y_test`) using `.score()`.

## Results
- The accuracy of the trained model on the test dataset is printed and can be further analyzed for performance improvement.

## Usage
- run commad (`pip install numpy pandas nltk sklearn`) to ensure all packages are installed
- Ensure the dataset (`spam_ham_dataset.csv`) is placed in the same directory or update the file path accordingly.
- Run the Python script to preprocess data, train the model, and evaluate its accuracy.

## Dependencies
- Python 3.x
- numpy
- pandas
- nltk
- scikit-learn

## Acknowledgements
- Dataset sourced from Kaggle: [Spam/ham email classification dataset](https://www.kaggle.com/datasets/venky73/spam-mails-dataset/data).

