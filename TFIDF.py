#TFIDF
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# 1. Load Dataset
# -------------------------------
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    df.columns = [col.strip().lower() for col in df.columns]
    
    # Keep original title
    df['original_title'] = df['title']
    
    return df

# -------------------------------
# 2. Clean Text (ONLY needed cols)
# -------------------------------
def clean_text(text):
    if pd.isnull(text):
        return ""
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', str(text))
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def clean_dataset(df):
    for col in ['title', 'overview', 'genres']:
        df[col] = df[col].apply(clean_text)
    return df

# -------------------------------
# 3. Feature Engineering (Weighted)
# -------------------------------
def combine_features(df):
    df['combined_features'] = (
        df['genres'] * 3 + " " +
        df['title'] * 2 + " " +
        df['overview']
    )
    return df

# -------------------------------
# 4. Vectorization
# -------------------------------
def vectorize_features(df):
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    return tfidf.fit_transform(df['combined_features'])

# -------------------------------
# 5. Similarity
# -------------------------------
def compute_similarity(tfidf_matrix):
    return cosine_similarity(tfidf_matrix)

# -------------------------------
# 6. Fuzzy Search (Better UX)
# -------------------------------
def find_closest_title(movie_name, df):
    movie_name = clean_text(movie_name)
    
    matches = df[df['title'].str.contains(movie_name)]
    
    if len(matches) > 0:
        return matches.index[0]
    
    return None

# -------------------------------
# 7. Recommendation
# -------------------------------
def recommend_movie(movie_name, df, similarity_matrix, top_n=5):
    
    idx = find_closest_title(movie_name, df)
    
    if idx is None:
        return "Movie not found. Try a different name."
    
    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    
    recommendations = []
    for i in scores[1:top_n+1]:
        recommendations.append(df.iloc[i[0]]['original_title'])
    
    return recommendations

# -------------------------------
# 8. Pipeline
# -------------------------------
def build_recommender(file_path):
    df = load_dataset(file_path)
    df = clean_dataset(df)
    df = combine_features(df)
    
    tfidf_matrix = vectorize_features(df)
    similarity_matrix = compute_similarity(tfidf_matrix)
    
    return df, similarity_matrix

# -------------------------------
# Usage
# -------------------------------
file_path = "/content/movies.csv"
df, similarity_matrix = build_recommender(file_path)

movie = "Crime 101"
print(recommend_movie(movie, df, similarity_matrix, top_n=5))

