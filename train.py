import pandas as pd
import sqlite3
import re
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization,
    LSTM, Bidirectional, TextVectorization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow.keras import layers
from nltk.corpus import stopwords
import argparse
import nltk
import json
import sklearn
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))

# function to remove stopwords
def remove_stopwords(text):
    word_tokens = word_tokenize(text)
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
    # Reconstruct the sentence
    return ' '.join(filtered_sentence)

def clean_text(text):
    # Remove HTML tags
    text = re.sub('<.*?>', '', text)
    text = re.sub("\'", "", text) 
    # remove everything except alphabets 
    text = re.sub("[^a-zA-Z]"," ",text) 
    # remove whitespaces 
    text = ' '.join(text.split()) 
    # convert text to lowercase 
    text = text.lower() 
    
    return text


def create_model(sqlite_file):
    # Connect to the SQLite database
    conn = sqlite3.connect(sqlite_file)
    df_tvmaze = pd.read_sql("select * from tvmaze", conn)
    df_tvmaze_genre = pd.read_sql("select * from tvmaze_genre", conn)
    # Add a 'value' column with all entries set to 1
    df_tvmaze_genre['value'] = 1

    # Use pivot_table to create a one-hot-encoded DataFrame
    pivot_encoded_df = df_tvmaze_genre.pivot_table(index='tvmaze_id', columns='genre', values='value', aggfunc='max').fillna(0).astype(int).reset_index()
    merged_df = pivot_encoded_df.merge(df_tvmaze[['tvmaze_id', 'description']], on='tvmaze_id', how='left')
    merged_df = merged_df[merged_df['description'].notna()]
    merged_df['description'] = merged_df['description'].apply(lambda x: clean_text(x))
    merged_df.reset_index(drop=True, inplace=True)
    
    #stop_words = set(stopwords.words('english'))
    
    # Removing Stopwords
    #merged_df['description'] = merged_df['description'].apply(lambda x: remove_stopwords(x))
    merged_df['description'] = merged_df['description'].apply(remove_stopwords)
    
    merged_df['word_count'] = merged_df['description'].apply(lambda x: len(x.split()))

    # Filter out rows where the word count is zero
    merged_df = merged_df[merged_df['word_count'] > 0]
    merged_df.drop(columns=['word_count'], inplace=True)
    
    genre_columns = merged_df.columns.drop(['tvmaze_id', 'description'])
    genre_col = list(genre_columns)
    with open('genre_columns.json', 'w') as f:
        json.dump(genre_col, f, ensure_ascii=False)

    target = merged_df[genre_columns].values

    tv_data, test_data, tv_labels, test_labels = sklearn.model_selection.train_test_split(merged_df.description, target)
    train_data, validation_data, train_labels, validation_labels = sklearn.model_selection.train_test_split(tv_data, tv_labels)
    
    # Tokenizing descriptions using a simple split on spaces
    desc_lengths = merged_df['description'].apply(lambda x: len(x.split()))

    # Checking statistics
    # print(desc_lengths.describe())
    percentile_90 = desc_lengths.quantile(0.9)
    percentile_95 = desc_lengths.quantile(0.95)
    percentile_99 = desc_lengths.quantile(0.99)
    
    num_genres = len(genre_columns)

    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(train_data)
    vocabulary_size = len(tokenizer.word_index)

    # max_tokens = vocabulary_size
    # output_sequence_length = int(percentile_95)
    # embedding_dim = 100

    max_tokens = vocabulary_size
    output_sequence_length = int(percentile_99)
    embedding_dim = 100

    vectorizer = TextVectorization(max_tokens=max_tokens, output_sequence_length=output_sequence_length)
    vectorizer.adapt(train_data)

    inputs = Input(shape=(1,), dtype=tf.string)
    x = vectorizer(inputs)
    x = Embedding(vocabulary_size + 1, embedding_dim)(x)
    x = GlobalAveragePooling1D()(x)
    # x = LSTM(128, return_sequences=False)(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.35)(x)
    outputs = Dense(num_genres, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

    history = model.fit(train_data, train_labels,
                        validation_data=(validation_data, validation_labels),
                        epochs=100,
                        callbacks=[early_stopping])
    
    loss, accuracy = model.evaluate(test_data, test_labels)
    print(f"Model training complete. Accuracy: {accuracy:.2f}")
    model.save('genre-model')
    print("Model Saved")
    conn.close()
    

# def train_model(data):
#     # Create and train a machine learning model
    
#     return model

# def evaluate_model(model, X_test, y_test):
#     # Make predictions on the test set
#     return accuracy

# def save_model(model):
#     # Save the trained model to paths.location_of_model
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a classification model from SQLite data")
    parser.add_argument("--training-data", required=True, help="Path to the SQLite database file")

    args = parser.parse_args()
    
    create_model(args.training_data)

    # # Load data from the SQLite database
    # data = load_data(args.training_data)

    # # Train the model
    #train_model(model)

    # # Evaluate the model (assuming you want to)
    # accuracy = evaluate_model(model, X_test, y_test)

    # print(f"Model training complete. Accuracy: {accuracy:.2f}")

    # Do you want to retrain the model on the whole data set now?

    # Save the trained model to a file
    # save_model(model)