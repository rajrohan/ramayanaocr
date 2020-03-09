import inspect
import re
from datetime import datetime
import pandas as pd
from tqdm import tqdm
from gensim.summarization import summarize
import networkx as nx
import math
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

seq = 1
log_columns = ["seq","timestamp","process","function","output"]
log_tbl = pd.DataFrame(columns = log_columns)

def create_str_from_list(original_read_text):
    # convert list of text as string 
    prepared_text = ""
    for line in original_read_text:
        line = line.split()
#       print(line)
        tmp_line = " ".join(line)
        prepared_text +=tmp_line
    return prepared_text

# Python code to get difference of two lists 
# Not using set() 
def diff(li1, li2): 
    li_dif = [i for i in li1 + li2 if i not in li1 or i not in li2] 
    return li_dif 


def clean_text(pattern, text):
    new_text = text
    new_text = re.sub(pattern, '', str(new_text)) # https://stackoverflow.com/questions/43727583/re-sub-erroring-with-expected-string-or-bytes-like-object
    return new_text


# logging and creation of regex_matched_list(list containing strings that match the pattern)
def whoami():
    return inspect.stack()[1][3]
def whosdaddy():
    return inspect.stack()[2][3]

def create_match_list_n_log(pattern,prepared_text):
    matched_list = re.findall(pattern,
                     prepared_text)
    len_match = len(matched_list)
    set_match = set(matched_list)
    output_tuple = len_match,set_match
#     print(output_tuple)
#     regex_list.append(set_match)
    
    global seq,log_tbl    
#     tbl(seq, datetime.now(),"parent: %s and child: %s" % (whosdaddy(),whoami()),pattern,output_tuple)
    to_append = [seq, datetime.now(),"parent: %s and child: %s" % (whosdaddy(),whoami()),pattern,output_tuple]
    a_series = pd.Series(to_append, index = log_tbl.columns)
    log_tbl = log_tbl.append(a_series, ignore_index=True)
    
    seq += 1
    return set_match


def store_file(path,obj):
    text_file = open(path, "w",encoding='utf-8')
    text_file.write(str(obj))
    text_file.close()
    return True

# remove regex matched text (for single pattern)
# def clean_text(rgxed_list, text):
#     new_text = text
#     if len(rgxed_list) > 0:
#         for rgx_match in rgxed_list:
# #             print(rgx_match, type(rgx_match))
#             new_text = re.sub(rgx_match, '', str(new_text)) # https://stackoverflow.com/questions/43727583/re-sub-erroring-with-expected-string-or-bytes-like-object
#     return new_text


def create_list_remove_number(text):
    text_sentence = []
    for word in tqdm(text):
        if re.findall("^\s?\d+\s?",word):
            continue
        else:   
            text_sentence.append(word)
    return text_sentence

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