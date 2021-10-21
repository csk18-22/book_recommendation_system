import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.preprocessing import MinMaxScaler
from skimage import io
import cv2
import base64

df = pd.read_csv('books.csv',error_bad_lines = False)
df = df.fillna(0)

# making copy of dataframe
df2 = df.copy()

df2.loc[ (df2['average_rating'] >= 0) & (df2['average_rating'] <= 1), 'rating_between'] = "between 0 and 1"
df2.loc[ (df2['average_rating'] > 1) & (df2['average_rating'] <= 2), 'rating_between'] = "between 1 and 2"
df2.loc[ (df2['average_rating'] > 2) & (df2['average_rating'] <= 3), 'rating_between'] = "between 2 and 3"
df2.loc[ (df2['average_rating'] > 3) & (df2['average_rating'] <= 4), 'rating_between'] = "between 3 and 4"
df2.loc[ (df2['average_rating'] > 4) & (df2['average_rating'] <= 5), 'rating_between'] = "between 4 and 5"

rating_df = pd.get_dummies(df2['rating_between'])
rating_df = rating_df.replace(np.nan, 0)

features = pd.concat([rating_df, df2['average_rating'], df2['ratings_count']], axis=1)

min_max_scaler = MinMaxScaler()
features = min_max_scaler.fit_transform(features)

# KNN model
model = neighbors.NearestNeighbors(n_neighbors=6, algorithm='auto')
model.fit(features)
dist, idlist = model.kneighbors(features)

st.title('Book Recommendation System')

# crop image function
def crop(img, center, width, height):
    return cv2.getRectSubPix(img, (width, height), center)

# main function
def BookRecommender(book_name):
    book_list_name = []
    dictionary = dict()

    cols = st.columns(3)

    book_id = df2[df2['title'] == book_name].index
    try:
        id = book_id[0]
    except:
        id = 0

    if (id == 0):
        st.write("Book not Found !! Please enter valid book name")
    else:
        book_id = book_id[0]
        ex = []
        i=0
        for newid in idlist[book_id]:

            book_list_name = []
            name = df2.iloc[newid]['title']
            url = df2.iloc[newid]['thumbnail']
            cols[i].write(name)

            if ( url != 0):
                image = io.imread(url)
                img = cv2.imshow('image',image)
                cropped = crop(image,center=(70, 85), width=150, height=240)
                cols[i].image(cropped)
                if(i==2):
                    i=0
                else:
                    i=i+1
            else:
                cols[i].image("imagenotfound.jpg",width=150)
                if(i==2):
                    i=0
                else:
                    i=i+1

# front end
st.markdown("""
<style>
body {
  color: red;
}
</style>
    """, unsafe_allow_html=True)

main_bg = "bgimg.jpg"
main_bg_ext = "jpg"

st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
        background-size:cover;
    }}

    </style>
    """,
    unsafe_allow_html=True
)

book_name = st.text_input("Book Name:", "")
BookNames = BookRecommender(book_name)
