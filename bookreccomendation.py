from re import A
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import base64
from skimage import io
import cv2
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('books1.csv',error_bad_lines = False)

# print(df.head())
df = df.fillna(0)
df['average_rating'] = df['average_rating'].fillna(0)
top_ten = df[df['ratings_count'] > 1000000]
top_ten.sort_values(by='average_rating', ascending=False)
plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(10, 10))
data = top_ten.sort_values(by='average_rating', ascending=False).head(10)
sns.barplot(x="average_rating", y="title", data=data, palette='inferno')



# most_books = df.groupby('authors')['title'].count().reset_index().sort_values('title', ascending=False).head(10).set_index('authors')
# plt.figure(figsize=(15,10))
# ax = sns.barplot(most_books['title'], most_books.index, palette='inferno')
# ax.set_title("Top 10 authors with most books")
# ax.set_xlabel("Total number of books")
# totals = []
# for i in ax.patches:
#     totals.append(i.get_width())
# total = sum(totals)
# for i in ax.patches:
#     ax.text(i.get_width()+.2, i.get_y()+.2,str(round(i.get_width())), fontsize=15,color='black')
# plt.show()



# most_rated = df.sort_values('ratings_count', ascending = False).head(10).set_index('title')
# plt.figure(figsize=(15,10))
# ax = sns.barplot(most_rated['ratings_count'], most_rated.index, palette = 'inferno')
# totals = []
# for i in ax.patches:
#     totals.append(i.get_width())
# total = sum(totals)
# for i in ax.patches:
#     ax.text(i.get_width()+.2, i.get_y()+.2,str(round(i.get_width())), fontsize=15,color='black')
# plt.show()




# df.average_rating = df.average_rating.astype(float)
# fig, ax = plt.subplots(figsize=[15,10])
# sns.distplot(df['average_rating'],ax=ax)
# ax.set_title('Average rating distribution for all books',fontsize=20)
# ax.set_xlabel('Average rating',fontsize=13)



# ax = sns.relplot(data=df, x="average_rating", y="ratings_count", color = 'red', sizes=(100, 200), height=7, marker='o')
# plt.title("Relation between Rating counts and Average Ratings",fontsize = 15)
# ax.set_axis_labels("Average Rating", "Ratings Count")


# plt.figure(figsize=(15,10))
# ax = sns.relplot(x="average_rating", y="num_pages", data = df, color = 'red',sizes=(100, 200), height=7, marker='o')
# ax.set_axis_labels("Average Rating", "Number of Pages")


main_bg = "sample2.jpg"
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


df2 = df.copy()
# print(df2.title)

df2.loc[ (df2['average_rating'] >= 0) & (df2['average_rating'] <= 1), 'rating_between'] = "between 0 and 1"
df2.loc[ (df2['average_rating'] > 1) & (df2['average_rating'] <= 2), 'rating_between'] = "between 1 and 2"
df2.loc[ (df2['average_rating'] > 2) & (df2['average_rating'] <= 3), 'rating_between'] = "between 2 and 3"
df2.loc[ (df2['average_rating'] > 3) & (df2['average_rating'] <= 4), 'rating_between'] = "between 3 and 4"
df2.loc[ (df2['average_rating'] > 4) & (df2['average_rating'] <= 5), 'rating_between'] = "between 4 and 5"

rating_df = pd.get_dummies(df2['rating_between'])

rating_df = rating_df.replace(np.nan, 0)

features = pd.concat([rating_df,
                      df2['average_rating'],
                      df2['ratings_count']], axis=1)

min_max_scaler = MinMaxScaler()
features = min_max_scaler.fit_transform(features)

model = neighbors.NearestNeighbors(n_neighbors=6, algorithm='ball_tree')
model.fit(features)
dist, idlist = model.kneighbors(features)

st.title('Book Recommendations !!! ')

def crop(img, center, width, height):
    return cv2.getRectSubPix(img, (width, height), center)

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
      st.write("Book not Found ! Sorry :(")
        # st.write("")
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

book_name = st.text_input("Book Name:", "")
BookNames = BookRecommender(book_name)
