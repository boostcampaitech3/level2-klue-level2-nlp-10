import pandas as pd


# 400자 이상의 data는 토큰 길이가 길고 max_token_len에 의해 짤릴 가능성이 높아 제거
# 제거 하는 경우와 안하는 경우 나눠서 실험해보면 좋을 듯
# def wordlen(text):
#     return len(text) < 400


    
def Preprocess(df : pd.DataFrame):

    """ 전처리 과정을 수행합니다.
    1) (Deprecated) 토큰화 과정에서 잘 되지 않은 400 글자 이상의 sentence data를 제거합니다. (400 글자 이상의 data를 포함하기로 합의하였음)
    2) word, sentence, subj_start_idx, obj_start_idx, label이 중복된 data를 제거합니다.
    3) 위의 경우에서 label만 다른 경우를 찾아 올바른 label에 대해 합의를 통해 결정하여 반대의 경우를 제거합니다.
    4) (Deprecated) Augmentation을 위해 BMP characters 이외의 문자를 space char로 치환합니다. (Augmentation은 제외하기로 합의하였음)
    Returns:
        Dataframe: 전처리가 완료된 Dataframe
    """    
    # df_v1=df[df['sentence'].apply(wordlen)] # 400 글자 이상 제거
    df_v2=df.drop_duplicates(['subj_word','sentence','obj_word','subj_start','obj_start','label'],keep='first') # word,sen,idx,label 중복된 경우 처음 등장한 경우를 제외하고 제거
    df_v3 = df_v2.drop([6749,8364,22258,277,25094]) # word,sen,idx가 동일하고, label만 다른 경우 이상 label data 제거
    
    # import re
    # only_BMP_pattern = re.compile("["
    #     u"\U00010000-\U0010FFFF"  #BMP characters 이외
    #                        "]+", flags=re.UNICODE)
    # df_v3 = df_v3.replace(to_replace = only_BMP_pattern, value ='', regex = True)
    return df_v3