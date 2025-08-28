import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import linear_kernel

tfidf = joblib.load('Model deploy/tfidf.pkl')     
vector = joblib.load('Model deploy/vector.pkl')    
final_df = pd.read_csv('Data/Final_data.csv')  
image = pd.read_csv('Data/Books.csv')

def recommend(book_name):
    try:
        book_index = final_df[final_df['Book-Title'] == book_name].index[0]
        similarities = linear_kernel(vector[book_index], vector).flatten()
        similar_books = similarities.argsort()[-6:-1][::-1]
        return [final_df.iloc[x]['Book-Title'] for x in similar_books]
    except IndexError:
        return ["Book not found in dataset"]
st.title("Book Recommendation System")
book_list = final_df['Book-Title'].values
selected_book = st.selectbox("Choose a book", book_list)

if st.button("Recommend"):
    recommendations = recommend(selected_book)
    for i in range(0, len(recommendations), 3):
        cols = st.columns(3)
        for j, col in enumerate(cols):
            if i + j < len(recommendations):
                book = recommendations[i + j]
                img_url = image[image['Book-Title'] == book]['Image-URL-L'].values[0]
                col.image(img_url, width=150)
                col.write(book)

# Streamlit run 'Model deploy/recommender.py'