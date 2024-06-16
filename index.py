import argparse
import sqlite3
import pandas as pd
import re
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

# Ensure that the required NLTK data is available
nltk.download('punkt')
nltk.download('stopwords')

def clean_text(text):
    # Function to clean text for processing
    text = re.sub('<.*?>', '', text)
    text = text.lower()
    text = re.sub('[^\w\s]', '', text)
    return text

#Week 10 Tokenizers
def tokenize_and_stem(text):
    stemmer = PorterStemmer()
    nltk_stop_words = stopwords.words('english')
    tokens = nltk.word_tokenize(text)
    stems = [stemmer.stem(token) for token in tokens if token not in nltk_stop_words]
    return stems

def create_search_index(db_path):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    df_tvmaze = pd.read_sql("SELECT * FROM tvmaze", conn)
    pd.to_pickle(df_tvmaze, 'df_tvmaze.pkl')
        

    # Process the DataFrame
    df_tvmaze['description'] = df_tvmaze['description'].fillna('')
    df_tvmaze['showname'] = df_tvmaze['showname'].fillna('')
    df_tvmaze['description'] = df_tvmaze['description'].apply(clean_text)
    df_tvmaze['showname'] = df_tvmaze['showname'].apply(clean_text)
    df_tvmaze['text_data'] = df_tvmaze.apply(lambda x: x['showname'] + ' ' + x['description'], axis=1)

    # Initialize TfidfVectorizer with the tokenizer and stemmer
    vectorizer = TfidfVectorizer(tokenizer=tokenize_and_stem, stop_words=None)
    tfidf_matrix = vectorizer.fit_transform(df_tvmaze['text_data'])

    # Save the vectorizer and matrix for later use
    pd.to_pickle(vectorizer, 'vectorizer.pkl')
    pd.to_pickle(tfidf_matrix, 'tfidf_matrix.pkl')

        
    print("Index created and saved successfully.")
    conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index data from an SQLite database")
    parser.add_argument("--raw-data", required=True, help="Path to the SQLite database file")

    args = parser.parse_args()

    create_search_index(args.raw_data) 

