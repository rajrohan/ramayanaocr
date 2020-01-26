import re
import os,glob
import pandas as pd
from datetime import datetime
import inspect

fulltext = []
regex_list = []

seq = 1

# Logging

log_columns = ["seq","timestamp","process","function","output"]
log_tbl = pd.DataFrame(columns = log_columns)

# Read all seprate text file to one list
folder_path = '../data/gt/txt/'
for filename in sorted(glob.glob(os.path.join(folder_path, '*.txt'))):
    with open(filename, 'r') as f:
        text = f.read()
        fulltext.append( text )


# convert list of text as string 
def create_str_from_list(original_read_text):
    prepared_text = ""
    for line in fulltext:
        line = line.split()
#       print(line)
        tmp_line = " ".join(line)
        prepared_text +=tmp_line
    return prepared_text

# remove regex matched text (for single pattern)
# def clean_text(rgxed_list, text):
#     new_text = text
#     if len(rgxed_list) > 0:
#         for rgx_match in rgxed_list:
# #             print(rgx_match, type(rgx_match))
#             new_text = re.sub(rgx_match, '', str(new_text)) # https://stackoverflow.com/questions/43727583/re-sub-erroring-with-expected-string-or-bytes-like-object
#     return new_text

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




# remove all types of regex patterns
def header_removal(prepared_text,pattern_list):
    for pattern in pattern_list:
        tmp_match_set = create_match_list_n_log(pattern,prepared_text)
#         print(len(tmp_match_set))
        regex_list.append(tmp_match_set)
    
#     print(len(regex_list))
    
    for pattern in pattern_list:
#         print(len(single_list))
        prepared_text = clean_text(pattern,prepared_text)
#         print(prepared_text[0:10])
    return prepared_text

prepared_text = create_str_from_list(fulltext)

regex_page_number = r"\d[०-९][०-९][०-९]*\s"
regex_shrivalmiki_ramayana = r"\w.\w.\w.ल्म.\w.\s\w.\w.\w."
regex_sundar_kand = r"\wन्दर\sका.|\w.न्दर\sकाण.\w+|\w.न्दर\sकाण्ड." #"\wन्दर\sका|\w.न्दर\sका"
regex_sarg = r"\wर्ग\s[०-९]*\W.[०-९]+|\wर्ग\s[०-९]*\W.[०-९]*" #"\wर्ग\s[०-९]*.|\wर्ग\s[०-९]*.[०-९]"

pattern_list = [regex_page_number,
                regex_sarg,
                regex_sundar_kand,
                regex_shrivalmiki_ramayana]

cleaned_text = header_removal(prepared_text,pattern_list)

def sanskrit_hindi_separate(prepared_text):
    
    resultHin = ''
    resultSan = ''
    muul_pattern = r"\w.ल-|\wल-"
    tika_pattern = r"\w.का-|\wका-"
    total_muul = re.findall(muul_pattern, prepared_text)
    for x in range(len(total_muul)):
        tmp_match_set = create_match_list_n_log(muul_pattern,prepared_text)
        regex_list.append(tmp_match_set)
        before_muul = re.split(muul_pattern,prepared_text,maxsplit=1)
        #print("muul",len(result1))
        #print ("muul",result1[0] )
        if len(before_muul)>1:
            tmp_match_set = create_match_list_n_log(tika_pattern,before_muul[1])
            regex_list.append(tmp_match_set)
            after_muul =re.split(tika_pattern,before_muul[1],maxsplit=1)
        resultHin += before_muul[0]
        resultSan += after_muul[0]
        #print("tika",len(result2))
        #print("tika",result2[0]) 
        if len(after_muul)>1:
            prepared_text = after_muul[1]
        else:
            prepared_text = after_muul[0]
            #print(x)
            break
    
    return resultHin,resultSan


resultHin,resultSan = sanskrit_hindi_separate(cleaned_text)

text_file = open("../data/hindi_text.txt", "w")
n = text_file.write(resultHin)
text_file.close()

text_file = open("../data/san_text.txt", "w")
n = text_file.write(resultSan)
text_file.close()