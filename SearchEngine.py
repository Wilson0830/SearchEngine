import pandas as pd
import numpy as np
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

##Read and Display CSV file data
FoodList = pd.read_csv("All_Grocery_and_Gourmet_Foods.csv")
#print(FoodList.head(1))

##Set the Maximum Width of Column in Database
pd.set_option('display.max_colwidth', None)
#print(FoodList.head(1))

##Check Duplicated Datas
#print(FoodList.duplicated().sum())

##Change the name to all lower case as keyword(Add new column)
FoodList['#keywords'] = FoodList['name'].str.lower()
#print(FoodList.head(1))

##Display total column in database
#print(FoodList.columns)

##Display total number of same/ different data in the specific column
#print(FoodList['main_category'].value_counts())
#print(FoodList['sub_category'].value_counts())
#print(FoodList.info())

##Remove specific columns from the database
FoodList.drop(['main_category','sub_category'], axis=1, inplace=True)
#print(FoodList.head(1))

##Preprocessing text
FoodList['#keywords']= FoodList['#keywords'].str.replace('''[^\w\d\s]''','',regex=True)
#print(FoodList['#keywords'])

##Streamer for reducing memory usage for processing the data
stemmer = PorterStemmer()
def stemming(text):
    words = []
    for word in text.split(' '):
        words.append(stemmer.stem(word))
    return ' '.join(words)

FoodList['#keywords'] = FoodList['#keywords'].apply(stemming)
#print(FoodList['#keywords'])

##Convert data into vector based on frequency of each word that occur in the entire text
cv = CountVectorizer(max_features = 5000, stop_words = 'english', dtype = np.uint8)
cv.fit(FoodList['#keywords'])
vector = cv.transform(FoodList['#keywords']).toarray()
#print(vector.shape)

##Determine the similarity between vectors (for search engine)
similarity = cosine_similarity(vector)

del(vector)
FoodList.drop(['#keywords'], axis=1, inplace=True)
#print(similarity.shape)
#print(similarity[0])

    ##Display random 10 sample from database
#print(FoodList['name'].sample(10,random_state=5))

    ##Example of search engine
#product= "Saffola Fittify Himalayan Apple Cider Vinegar with the Mother of Vinegar - 500 ml"
#print(FoodList[FoodList['name']== product])

##Display recommendation of 10 product based on searched text
def recommender(product):
    product_index = FoodList[FoodList['name']== product].index[0]
    similarity_list= list(enumerate(similarity[product_index]))
    top_10_similar_product = sorted(similarity_list, key=lambda x: x[1], reverse =True)[1:11]
    for idx, similary in top_10_similar_product:
        print(FoodList.loc[idx]['name'])

#recommender("Saffola Fittify Himalayan Apple Cider Vinegar with the Mother of Vinegar - 500 ml")

##Converting python object into a byte or stream to store in database
pickle.dump(similarity, open('similarity.pkl', 'wb'))
pickle.dump(FoodList, open('data.pkl', 'wb'))
