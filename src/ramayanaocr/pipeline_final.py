#!/usr/bin/env python
# coding: utf-8

# In[2]:


# https://www.kaggle.com/kernels/scriptcontent/11511967/notebook

import string
import numpy as np
import pandas as pd
# from IPython.display import display
from tqdm import tqdm
from collections import Counter
import ast
import re
from pprint import pprint

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
# import matplotlib.mlab as mlab
import seaborn as sb

from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
# from textblob import TextBlob
import scipy.stats as stats

from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE

import nltk
from nltk.tag import tnt
from nltk.corpus import stopwords
from nltk.corpus import indian
from nltk.tokenize import word_tokenize,sent_tokenize

import gensim
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import gensim.corpora as corpora
from gensim.summarization import summarize
from gensim.test.utils import datapath

import stanfordnlp
import networkx as nx
import math

import warnings
warnings.filterwarnings("ignore")

# from bokeh.plotting import figure, output_file, show
# from bokeh.models import Label
# from bokeh.io import output_notebook
# output_notebook()

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


nlp = stanfordnlp.Pipeline ( lang = 'hi' ) 


# In[3]:


datafile = '../data/hindi.txt'
with open(datafile,'r',encoding='utf-8') as f:
    text = f.read()
    text = text.split("॥")


# In[179]:


text_sentence = []
i = 0
for word in tqdm(text):
#     print(i,word)
    if re.findall("^\s?\d+\s?",word):
        continue
    else:   
        text_sentence.append(word)


# In[180]:


text_sentence[0:2]


# In[6]:


len(text_sentence) #total sentences


# In[7]:


# hindi stopwords
stop_words_df = pd.read_csv("../data/stopwords.txt", header = None)
stop_words = list(set(stop_words_df.values.reshape(1,-1).tolist()[0]))
stop_words.extend(["।", "।।", ")", "(", ",",'"',"हे", "हो", 'में','से','COMMA'])


# In[8]:


len(stop_words)


# In[10]:


stop_words[:2]


# In[11]:


def create_str_from_list(original_read_text):
    prepared_text = ""
    for line in original_read_text:
        line = line.split()
#       print(line)
        tmp_line = " ".join(line)
        prepared_text += " \n"+tmp_line
    return prepared_text


# In[12]:


str_text = create_str_from_list(text_sentence)


# In[13]:


str_text[0:2500]


# In[14]:


summarize(str_text[0:2500])


# In[19]:


len(str_text)


# In[17]:


df = pd.DataFrame()


# In[67]:


def summarization(str_text):
    startflag = 0
    endflag = 2500
    parsed_text = {'text':[], 'title':[]}
    for summary in range(int(len(str_text)/endflag)):
        summarization_block = summarize(str_text[startflag:endflag])
        parsed_text['text'].append(str_text[startflag:endflag])
        parsed_text['title'].append(summarization_block)
        startflag +=2500
        endflag +=2500
    return pd.DataFrame(parsed_text)


# In[68]:


df = summarization(str_text)
df.shape


# In[69]:


#df.to_csv('input_abstarct_summary.csv')


# ### Alternative Summarization from Gensim 

# In[220]:


def textrankTfIdf(document):
    # sentence_tokenizer = PunktSentenceTokenizer()
    # sentences = sentence_tokenizer.tokenize(document, 'hindi')

    sentences = document
    bow_matrix = CountVectorizer().fit_transform(sentences)
    # normalized = TfidfTransformer(norm='l2', use_idf=True, use_bm25idf=True, smooth_idf=True,
    #              delta_idf=False, sublinear_tf=False, bm25_tf=True).fit_transform(bow_matrix)

    normalized = TfidfTransformer().fit_transform(bow_matrix)
    similarity_graph = normalized * normalized.T

    nx_graph = nx.from_scipy_sparse_matrix(similarity_graph)
    scores = nx.pagerank(nx_graph)
    return sorted(((scores[i], s) for i, s in enumerate(sentences)),
                  reverse=True)
def orderSentences(rankedList, data, initSentences):
    index = ['']*len(data)
    # print(rankedList)
    for eachRanked in rankedList[0:int(math.ceil(0.2*len(rankedList)))]:
        sen = eachRanked[1]
        index[data.index(sen)] = initSentences[data.index(sen)]
        # print(data.index(sen))
    return index


# In[221]:


int(len(text_sentence)/25)


# In[222]:


def alternate_summarization(text_sentence):
    startflag = 0
    endflag = 25
    parsed_text = {'text':[], 'alt_title':[]}
    sentence =''
    for summary in range(int(len(text_sentence)/endflag)):
        rankedSentences = textrankTfIdf(text_sentence[startflag:endflag])
        orderedsentences = orderSentences(rankedSentences, text_sentence[startflag:endflag], text_sentence[startflag:endflag])
        for ordered in orderedsentences:
            if ordered != "":
                sentence += ordered 
        parsed_text['alt_title'].append(sentence)
        parsed_text['text'].append(text_sentence[startflag:endflag])
        startflag +=25
        endflag +=25
        sentence =''
    return pd.DataFrame(parsed_text)


# In[223]:


df2 = alternate_summarization(text_sentence)


# In[224]:


df2.head()


# In[226]:


#df2.to_csv('summary_alt_gensim.csv')


# In[4]:


#df2 = pd.read_csv('../data/summary_alt_gensim.csv')


# In[108]:


def extract_ner(sentence_ner,row_no):
    parsed_text = {'sentence':[], 'word':[], 'ner':[]}
    words = re.findall("([^<]+)<[^>]+>",sentence_ner)
    tags = re.findall("<([^>]+)>",sentence_ner)
    
    # for word in words: append word #we need to write seperate tags
    for index in range(len(words)):
        parsed_text['word'].append(words[index])
        parsed_text['ner'].append(tags[index])
        parsed_text['sentence'].append(row_no)
    
    return pd.DataFrame(parsed_text)


# In[146]:


from flair.data import Sentence
from flair.models import SequenceTagger
# load the model you trained
model = SequenceTagger.load('/Users/rohanraj/Documents/Master/CaseStudy2/code/resources/taggers/netag/best-model.pt')

df_ner = pd.DataFrame()

r, c = df2.shape
for row_no in range(r):
    sentence = Sentence(df2['alt_title'][row_no])
    model.predict(sentence)
    sentence_ner = sentence.to_tagged_string()
    temp_df = extract_ner(sentence_ner,row_no)
    df_ner = df_ner.append(temp_df,ignore_index=True)  
# predict tags and print
#print(sentence.to_tagged_string())


# In[151]:


df_ner.head()


# In[148]:


#df_ner.to_csv('../data/ner_tag.csv')


# In[154]:


#df_ner = pd.read_csv('../data/ner_tag.csv',index_col=0)


# In[155]:


df_ner.head()


# In[156]:


# Get names of indexes for which column Age has value 30
df_ner_imp = df_ner[ df_ner['ner'].str.endswith('PERSON') | df_ner['ner'].str.endswith('LOCATION') ]
# Delete these row indexes from dataFrame
#df_ner.drop(indexNames , inplace=True)


# In[159]:


df_ner_imp.head()


# In[158]:


#df_ner_imp.to_csv('../data/ner_tag_imp.csv')


# In[73]:


#df_ner.to_csv('../data/ner_tag.csv')


# In[73]:


# Standford nlp is being used for lemmatization


# In[95]:


def lemmatization(text):
    lemmatized_text = []
    for line in tqdm(text):
        if line not in [""," "] :
            doc = nlp(line)
            for sent in doc.sentences:
                for wrd in sent.words:
                    #extract text and lemma
                    lemmatized_text.append(wrd.lemma)
    return lemmatized_text

def remove_stopwords(word_tokenized,stop_words):
    return [word for word in word_tokenized if word not in stop_words]

def custom_remove_garbage(original_words_list,list_of_garbage_words):
    tmp_list = [word for word in original_words_list if word not in list_of_garbage_words] # garbage list
    tmp_list = [word for word in tmp_list if len(re.findall("\d+",word))==0] # english numbers
    tmp_list = [word for word in tmp_list if len(re.findall("[a-zA-Z]+",word))==0] # english alphabets
    return tmp_list

def Diff(li1, li2): 
    li_dif = [i for i in li1 + li2 if i not in li1 or i not in li2] 
    return li_dif 


# ## Topic Meddling after prepocessing

# In[149]:


def topic_modelling(text_sentence,stop_words):
    df = pd.DataFrame()
    startflag = 0
    endflag = 25
    for text in range(int(len(text_sentence)/endflag)):
        
        #Clean text after lemmatization
        lemmatized = lemmatization(text_sentence[startflag:endflag])
        clean_text = remove_stopwords(lemmatized,stop_words)
        #len(clean_text),len(lemmatized) #281,612
        
        # remove punctuation from each word
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in clean_text]
        #print(stripped[:25])
        #print(len(clean_text),len(stripped)) #281

        resultant = Diff(clean_text,stripped)
        #len(set(resultant)) #19

        final_text = custom_remove_garbage(stripped,resultant)
        #len(final_text) #265
        
        # Create Dictionary
        id2word = corpora.Dictionary([final_text])

        # Create Corpus
        texts = final_text

        # Term Document Frequency
        corpus = [id2word.doc2bow(texts)] 
        
        # Build LDA model

        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=1, 
                                           random_state=100,
                                           update_every=10,
                                           chunksize=100,
                                           passes=20,
                                           alpha='auto',
                                           per_word_topics=True)
        #pprint(lda_model.print_topics())
        temp_file = datapath("model")
        lda_model.save(temp_file)
        lda = gensim.models.ldamodel.LdaModel.load(temp_file)
        lda.update(corpus)
        lda.save('model')
        temp_df = pd.DataFrame(lda.print_topics())
        df = df.append(temp_df,ignore_index=True) 
        startflag +=25
        endflag +=25
    return df


# In[150]:


df4 = topic_modelling(text_sentence,stop_words)


# In[151]:


df4.shape


# In[152]:


df4.tail(5)


# In[153]:


#df4.to_csv('combined_model_topic.csv')


# In[160]:


temp_file = datapath("model")
lda = gensim.models.ldamodel.LdaModel.load(temp_file)


# In[3]:


#df4 =pd.read_csv('../data/combined_model_topic.csv',index_col=0)


# In[4]:


df4.head()


# In[186]:


#re.findall('[!@#$%^&*(),.?":{}|<>]',df4['1'][0])


# In[5]:


#re.sub('[!@#$%^&*(),.?"+:{}|<>0-9]', '', df4['1'][0])


# In[184]:


df4['1'][0]


# In[32]:


#df4.iloc[0]


# In[26]:


def process_topic(df):
    parsed_text = {'top_topic':[]}
    
    for ind in df.index:
        parsed_text['top_topic'].append(re.sub('[!@#$%^&*(),.?"+:{}|<>0-9]', '', df['1'][ind]))
    return pd.DataFrame(parsed_text)    


# In[27]:


topic_df= process_topic(df4)


# In[28]:


topic_df


# In[29]:


topic_df.to_csv('combined_model_topic_top_clean.csv')


# In[ ]:





# In[ ]:




