import pandas as pd
import re

# 400자 이상의 data는 토큰 길이가 길고 max_token_len에 의해 짤릴 가능성이 높아 제거
# 제거 하는 경우와 안하는 경우 나눠서 실험해보면 좋을 듯
def wordlen(text):
    return len(text) < 400


    
def Preprocess(df):
    # df_v1=df[df['sentence'].apply(wordlen)] # 400 글자 이상 제거
    df_v2=df.drop_duplicates(['subj_word','sentence','obj_word','subj_start','obj_start','label'],keep='first') # word,sen,idx,label 중복된 경우 제거
    df_v3 = df_v2.drop([6749,8364,22258,277,25094]) # word,sen,idx가 동일하고, label만 다른 경우 이상 label data 제거
    
    # only_BMP_pattern = re.compile("["
    #     u"\U00010000-\U0010FFFF"  #BMP characters 이외
    #                        "]+", flags=re.UNICODE)
    # df_v3 = df_v3.replace(to_replace = only_BMP_pattern, value ='', regex = True)
    # 현재 이상하게 preporcess에 넣으려니 계속 오류가 발생해서 train.py에 이 부분을 추가해야 할 듯.
    return df_v3