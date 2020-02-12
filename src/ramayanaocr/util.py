
# convert list to string
def create_str_from_list(original_read_text):
    prepared_text = ""
    for line in original_read_text:
        line = line.split()
#       print(line)
        tmp_line = " ".join(line)
        prepared_text += " "+tmp_line
    return prepared_text

# Python code to get difference of two lists 
# Not using set() 
def diff(li1, li2): 
    li_dif = [i for i in li1 + li2 if i not in li1 or i not in li2] 
    return li_dif 
