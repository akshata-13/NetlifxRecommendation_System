import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_data():
    df = pd.read_csv('netflix_titles.csv')
    df.fillna('', inplace=True)
    df['combined'] = df['title'] + ' ' + df['listed_in'] + ' ' + df['description']
    return df

@st.cache_data
def get_embeddings(data, _model):
    return _model.encode(data['combined'].tolist(), show_progress_bar=True)


# Load components
model = load_model()
df = load_data()
embeddings = get_embeddings(df, model)


# Create reverse title index
title_to_index = pd.Series(df.index, index=df['title'].str.lower()).drop_duplicates()

def recommend(title, top_n=10):
    title = title.lower()
    if title not in title_to_index:
        return []
    
    idx = title_to_index[title]
    sim_scores = cosine_similarity([embeddings[idx]], embeddings)[0]
    top_indices = sim_scores.argsort()[::-1][1:top_n+1]
    return df['title'].iloc[top_indices].tolist()

# Streamlit UI
st.title("🎬 Netflix Series Recommendation App")
user_input = st.text_input("Enter a Netflix show or movie title:")

if user_input:
    st.write(f"Top recommendations similar to **{user_input}**:")
    results = recommend(user_input)
    if results:
        for i, r in enumerate(results, 1):
            st.write(f"{i}. {r}")
    else:
        st.warning("Title not found in dataset.")
