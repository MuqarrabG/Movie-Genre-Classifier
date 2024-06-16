import argparse
import json
import os
import tensorflow as tf
import lime.lime_text
import pandas as pd
import numpy as np


def classify_tv_show(input_file, output_json_file, encoding='UTF-8', explanation_output_dir=None):
    try:
        # Read the description from the input file and genre columns
        with open('genre_columns.json', 'r') as f:
            genre_columns = json.load(f)
    
        with open(input_file, encoding=encoding) as file:
            description = file.read()

        # Load your model, perhaps from paths.location_of_model
        model = tf.keras.models.load_model('genre-model')
        
        # Implement your classification logic here to identify TV show genres
        text_explainer = lime.lime_text.LimeTextExplainer(class_names=genre_columns)
        # load your model from somewhere in the /app directory
        genre_dict = {genre: i for i, genre in enumerate(genre_columns)}    
        predictions = model.predict([description])

        predicted_class = np.argmax(predictions)
        # Extract the probabilities for each genre
        genres_probabilities = [(genre, prob) for genre, prob in zip(genre_columns, predictions[0])]

        # Get genres with probability greater than 0.5
        high_prob_genres = [genre for genre in genres_probabilities if genre[1] >= 0.5]

        # If there are no genres with probability greater than 0.5
        if not high_prob_genres:
            high_prob_genres = [genre for genre in genres_probabilities if genre[1] >= 0.3]

        # Sort genres by probability in descending order and select top 3
        top_3_genres = sorted(high_prob_genres, key=lambda x: x[1], reverse=True)[:3]

        # Example genres (replace with your actual genre prediction)
        genres = [genre[0] for genre in top_3_genres]

        # Write the identified genres to the output JSON file
        with open(output_json_file, 'w', encoding=encoding) as json_file:
            json.dump(genres, json_file, ensure_ascii=False)

        # Optionally, write an explanation to the explanation output directory
        if explanation_output_dir:
            explanation  = text_explainer.explain_instance(description, model.predict, labels=[predicted_class])
            explanation_filename = os.path.join(explanation_output_dir, "muq-model-explanation.html")
            explanation.save_to_file(explanation_filename)
            # explanation = "This is an example explanation."
            # with open(explanation_filename, 'w', encoding='UTF-8') as exp_file:
            #     exp_file.write(explanation)
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify TV show genres based on description")
    parser.add_argument("--input-file", required=True, help="Path to the input file with TV show description")
    parser.add_argument("--output-json-file", required=True, help="Path to the output JSON file for genres")
    parser.add_argument("--encoding", default="UTF-8", help="Input file encoding (default: UTF-8)")
    parser.add_argument("--explanation-output-dir", help="Directory for explanation output")

    args = parser.parse_args()

    classify_tv_show(args.input_file, args.output_json_file, args.encoding, args.explanation_output_dir)
