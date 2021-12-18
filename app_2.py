import pandas as pd
import numpy as np
import streamlit as st

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity


st.title('Product Recommendation App 2')
st.subheader('Choose a product from this table')
df = pd.read_csv("skindataall.csv")

display_df = df[['Brand', 'Product', 'Product_Url', 'Price', 'Rating']]
st.dataframe(display_df.sort_values('Product', ascending=False).drop_duplicates('Product').sort_index())

df_cont = df[['Product', 'Product_id', 'Ingredients', 'Product_Url', 'Ing_Tfidf', 'Rating', 'Price']]
df_cont.drop_duplicates(inplace=True)


tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(df_cont['Ingredients'])


cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


df_cont = df_cont.reset_index(drop=True)
titles = df_cont[['Product', 'Ing_Tfidf', 'Rating', 'Price']]
indices = pd.Series(df_cont.index, index=df_cont['Product'])

# Cache for a day
@st.cache(ttl=3600*24, show_spinner=False)

def content_recommendations(product):
    idx = indices[product]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    product_indices = [i[0] for i in sim_scores]
    return titles.iloc[product_indices]

search_term = st.text_input("Search product")
if st.button("Recommend"):
    if search_term is not None:
        try:
            results = content_recommendations(search_term)
            st.write('Based on your search, these are the top products for you:')
            st.write(results)

        except:
            results= "Not Found"
            st.warning(results)
