import re
import os,glob
import pandas as pd
from datetime import datetime
import inspect
from helper import create_str_from_list,create_match_list_n_log,clean_text,store_file

fulltext = []
regex_list = []

#seq = 1

# Logging

# log_columns = ["seq","timestamp","process","function","output"]
# log_tbl = pd.DataFrame(columns = log_columns)


def load_input_txt(path):
    # Read all seprate text file to one list
    folder_path = path
    for filename in sorted(glob.glob(os.path.join(folder_path, '*.txt'))):
        with open(filename, 'r',encoding='utf-8') as f:
            text = f.read()
            fulltext.append( text )
    return fulltext

def header_removal(fulltext):
    # convert List to string
    prepared_text= create_str_from_list(fulltext)
    # remove all types of regex patterns
    regex_page_number = r"\d[०-९][०-९][०-९]*\s"
    regex_shrivalmiki_ramayana = r"\w.\w.\w.ल्म.\w.\s\w.\w.\w."
    regex_sundar_kand = r"\wन्दर\sका.|\w.न्दर\sकाण.\w+|\w.न्दर\sकाण्ड." #"\wन्दर\sका|\w.न्दर\sका"
    regex_sarg = r"\wर्ग\s[०-९]*\W.[०-९]+|\wर्ग\s[०-९]*\W.[०-९]*" #"\wर्ग\s[०-९]*.|\wर्ग\s[०-९]*.[०-९]"

    pattern_list = [regex_page_number,
                    regex_sarg,
                    regex_sundar_kand,
                    regex_shrivalmiki_ramayana]
    for pattern in pattern_list:
        tmp_match_set = create_match_list_n_log(pattern,prepared_text)
         #print(len(tmp_match_set))
        regex_list.append(tmp_match_set)
    
     #print(len(regex_list))
    
    for pattern in pattern_list:
         #print(len(single_list))
        prepared_text = clean_text(pattern,prepared_text)
         #print(prepared_text[0:10])
    return prepared_text

def keep_hindi(prepared_text):
    
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
    return resultHin

fulltext = load_input_txt('./data/input/')
cleaned_text = header_removal(fulltext)
resultHin = keep_hindi(cleaned_text)
store_file("./data/output/hindi.txt",resultHin)

# now = datetime.now()
# current_time = now.strftime("%Y-%m-%d%S")
# filename = current_time+".log"
# path = "./data/log/"
# log_tbl.to_csv(path+filename,index=False)