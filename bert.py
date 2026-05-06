# TRAINING SCRIPT (RUN ONCE)
# -------------------------------

import pandas as pd
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# -------------------------------
# Cleaning Functions
# -------------------------------
def clean_text(text):
    if pd.isnull(text):
        return ""
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', str(text))
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

# -------------------------------
# Load + Prepare Data
# -------------------------------
def prepare_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = [col.strip().lower() for col in df.columns]

    df['original_title'] = df['title']

    for col in ['title', 'overview', 'genres']:
        df[col] = df[col].fillna("").apply(clean_text)

    df['combined_features'] = (
        df['genres'] * 3 + " " +
        df['title'] * 2 + " " +
        df['overview']
    )

    return df

# -------------------------------
# Train & Save
# -------------------------------
def train_and_save(file_path):

    df = prepare_data(file_path)

    print("🔄 Loading BERT model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("🔄 Encoding movies (one-time process)...")
    embeddings = model.encode(
        df['combined_features'].tolist(),
        show_progress_bar=True
    )

    print("🔄 Computing similarity matrix...")
    similarity_matrix = cosine_similarity(embeddings)

    # Save everything
    df.to_pickle("movies.pkl")
    np.save("similarity.npy", similarity_matrix)

    print("✅ Training completed and saved!")

# -------------------------------
# RUN THIS ONLY ONCE
# -------------------------------
if __name__ == "__main__":
    train_and_save("/content/movies.csv")







# INFERENCE SCRIPT (FAST)
# -------------------------------

import pandas as pd
import numpy as np
import re

# -------------------------------
# Cleaning Function
# -------------------------------
def clean_text(text):
    if pd.isnull(text):
        return ""
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', str(text))
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

# -------------------------------
# Load Saved Model
# -------------------------------
def load_model():
    df = pd.read_pickle("movies.pkl")
    similarity_matrix = np.load("similarity.npy")
    return df, similarity_matrix

# -------------------------------
# Find Movie (Partial Match)
# -------------------------------
def find_movie_index(movie_name, df):
    movie_name = clean_text(movie_name)

    matches = df[df['title'].str.contains(movie_name)]

    if len(matches) > 0:
        return matches.index[0]

    return None

# -------------------------------
# Recommendation Function
# -------------------------------
def recommend_movie(movie_name, df, similarity_matrix, top_n=5):

    idx = find_movie_index(movie_name, df)

    if idx is None:
        return "Movie not found. Try another name."

    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    recommendations = []
    for i in scores[1:top_n+1]:
        recommendations.append(df.iloc[i[0]]['original_title'])

    return recommendations

# -------------------------------
# RUN INTERACTIVE MODE
# -------------------------------
if __name__ == "__main__":

    print("⚡ Loading saved recommender...")
    df, similarity_matrix = load_model()

    print("✅ Ready! Get instant recommendations.\n")

    while True:
        movie = input("Enter movie name (or 'exit'): ")

        if movie.lower() == "exit":
            print("👋 Exiting...")
            break

        results = recommend_movie(movie, df, similarity_matrix)

        if isinstance(results, list):
            print("\n🎬 Recommendations:")
            for i, r in enumerate(results, 1):
                print(f"{i}. {r}")
        else:
            print(results)

        print("-" * 40)
