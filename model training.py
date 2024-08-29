#!/usr/bin/env python
# coding: utf-8

# In[165]:


import pandas as pd


# In[166]:


df=pd.read_csv("spotify_millsongdata.csv")


# In[167]:


df.head(10)


# In[168]:


df.tail(10
)


# In[169]:


df.shape


# In[170]:


df.isnull().sum()


# In[171]:


df.head(1)


# In[195]:


df= df.sample(5000).drop("link",axis=1).reset_index(drop=True)


# In[175]:


#df=df.drop("link",axis=1)


# In[176]:


df.head(10)


# In[198]:


df.tail(5)


# In[177]:


df.shape


# text cleaning/preprocessing

# In[178]:


df['text']= df['text'].str.lower().replace(r'^\w\s', ' ').replace(r'\n', ' ',regex= True)


# In[179]:


df.tail(5)


# In[ ]:





# In[180]:


import nltk
from nltk.stem.porter import PorterStemmer
nltk.download('punkt')


# In[181]:


stemmer= PorterStemmer()


# In[182]:


def token(txt):
    token= nltk.word_tokenize(txt)
    a=[stemmer.stem(w) for w in token]
    return " ".join(a)


# In[183]:


df['text'].apply(lambda x: token(x))


# In[184]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[185]:


tfid= TfidfVectorizer(analyzer='word', stop_words='english')


# In[186]:


tfid


# In[187]:


matrix= tfid.fit_transform(df['text'])


# In[188]:


matrix


# In[189]:


similar=cosine_similarity(matrix)


# In[190]:


similar[0]


# In[191]:


df[df['song']=="Ain't She Sweet"].index[0]


# Recommender Function

# In[203]:


def recommender(song_name):
    index=df[df['song']==song_name].index[0]
    dist=sorted(list(enumerate(similar[index])), reverse= True, key= lambda x:x[1])
    song=[]
    for sid in dist[1:15]:
        song.append(df.iloc[sid[0]].song)
    return song
        
    
    


# In[204]:


recommender("Dog Door")


# In[199]:


import pickle


# In[200]:


pickle.dump(similar, open("similarity","wb") )


# In[202]:


pickle.dump(df, open("df","wb"))


# In[ ]:




