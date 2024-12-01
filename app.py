import streamlit as st
import pandas as pd
import numpy as np
import pickle

FoodList = pickle.load(open('data.pkl','rb'))
similarity = pickle.load(open('similarity.pkl','rb'))

def recommendation(product):
    #Find the product in FoodList and store in FoodList_index
    FoodList_index = FoodList[FoodList['name']==product].index[0]
    #Pass the FoodList_index into similarity_list
    similarity_list = list(enumerate(similarity[FoodList_index]))
    #Sort the similarity_list into top 10 recommend product
    top_10_similar_product= sorted(similarity_list, key= lambda x : x[1], reverse = True) [1:11]

    similar_product=[]
    for idx, similar_score in top_10_similar_product:
        product_info =\
        {
            'name': FoodList.loc[idx]['name'],
            'image': FoodList.loc[idx]['image'],
        }
        similar_product.append(product_info)
    return similar_product

st.title('Product Search Engine Usign Python')

product_name = st.selectbox('Select a product:', FoodList['name'])

if st.button('Search'):
    search_product = recommendation(product_name)
    st.write(f"Top 10 recommended for -> {product_name}")

    for rec in search_product:
        st.image(rec['image'],width=150)
        st.write(f"**{rec['name']}**")

