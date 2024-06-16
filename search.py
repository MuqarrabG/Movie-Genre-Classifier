import argparse
import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import nltk

# Ensure that the required NLTK data is available
nltk.download('punkt')

def tokenize_and_stem(text):
    stemmer = nltk.stem.porter.PorterStemmer()
    tokens = nltk.word_tokenize(text)
    return [stemmer.stem(t) for t in tokens]

def perform_search(query, top_n=3):
    # Load the original dataframe
    df_tvmaze = pd.read_pickle('df_tvmaze.pkl')
    
    # Exact show name match
    exact_match = df_tvmaze[df_tvmaze['showname'].str.lower() == query.lower()]
    if not exact_match.empty:
        return exact_match[['tvmaze_id', 'showname']].to_dict(orient='records')
    
    # Load the vectorizer and the tfidf matrix
    vectorizer = pd.read_pickle('vectorizer.pkl')
    tfidf_matrix = pd.read_pickle('tfidf_matrix.pkl')

    # Process the query
    query_stemmed = " ".join(tokenize_and_stem(query))
    query_vec = vectorizer.transform([query_stemmed])

    # Perform the search
    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    related_docs_indices = cosine_similarities.argsort()[:-top_n-1:-1]

    # Get the results
    results = df_tvmaze.iloc[related_docs_indices][['tvmaze_id', 'showname']]
    
    return results.to_dict(orient='records')

def search_tv_shows(input_file, output_json_file, encoding='UTF-8'):
    try:
        # Read the search query from the input file
        with open(input_file, 'r', encoding=encoding) as file:
            search_query = file.read().strip()

        # Perform the search and get the matched shows
        matched_shows = perform_search(search_query)

        # Write the matched shows to the output JSON file
        with open(output_json_file, 'w', encoding=encoding) as json_file:
            json.dump(matched_shows, json_file, ensure_ascii=False, indent=4)

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search for TV shows based on a query")
    parser.add_argument("--input-file", required=True, help="Path to the input file with the search query")
    parser.add_argument("--output-json-file", required=True, help="Path to the output JSON file for matched shows")
    parser.add_argument("--encoding", default="UTF-8", help="Encoding for reading the input file (default: UTF-8)")
    args = parser.parse_args()

    search_tv_shows(args.input_file, args.output_json_file, args.encoding)
