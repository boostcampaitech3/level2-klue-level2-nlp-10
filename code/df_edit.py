import pandas as pd

def better_df(df : pd.DataFrame, mode : int):
    """ 기존에 주어진 CSV 파일을 개선합니다.
    1) subj, obj entity를 분해하여 type, word, start_idx, end_idx를 각각 담습니다.
    2) Train(mode = 0)의 경우 Preprocess를 위해 Labelnum을 부착합니다.
    3) Test(mode = 1)의 경우 Label이 임의의 숫자이기 때문에 Labelnum을 부착하지 않습니다.

    Args:
        df(Dataframe): read_csv를 통해 불러온 기존 Dataframe

    Returns:
        Dataframe->Dataframe: 개선된 Dataframe
    """    
    sub_type = []
    sub_word = []
    sub_start = []
    sub_end = []
    obj_type = []
    obj_word = []
    obj_start = []
    obj_end = []
    id = []

    # Subject, Object entity 풀어서 각종 작업에 용이하도록 변경
    for i in range(df.shape[0]):
        id.append((df['id'].iloc[i]))

        sub_type.append((eval(df['subject_entity'].iloc[i]))["type"])
        sub_word.append((eval(df['subject_entity'].iloc[i]))["word"])
        sub_start.append((eval(df['subject_entity'].iloc[i]))["start_idx"])
        sub_end.append((eval(df['subject_entity'].iloc[i]))["end_idx"])

        obj_type.append((eval(df['object_entity'].iloc[i]))["type"])
        obj_word.append((eval(df['object_entity'].iloc[i]))["word"])
        obj_start.append((eval(df['object_entity'].iloc[i]))["start_idx"])
        obj_end.append((eval(df['object_entity'].iloc[i]))["end_idx"])

    df_entity = pd.DataFrame(id)

    df_entity.columns = ['id']

    df_entity['subj_type'] = sub_type
    df_entity['subj_word'] = sub_word
    df_entity['subj_start'] = sub_start
    df_entity['subj_end'] = sub_end

    df_entity['subj_type'] = sub_type
    df_entity['subj_word'] = sub_word
    df_entity['subj_start'] = sub_start
    df_entity['subj_end'] = sub_end

    df_entity['obj_type'] = obj_type
    df_entity['obj_word'] = obj_word
    df_entity['obj_start'] = obj_start
    df_entity['obj_end'] = obj_end
    df_entity['sentence'] = df['sentence']
    df_entity['label'] = df['label']

    if mode == 0:
        # Preprocess 용 label_num 부착
        label_info = {'no_relation': 0, 'org:top_members/employees': 1, 'org:members': 2, 'org:product': 3, 'per:title': 4, 'org:alternate_names': 5, 'per:employee_of': 6, 'org:place_of_headquarters': 7, 'per:product': 8, 'org:number_of_employees/members': 9, 'per:children': 10, 'per:place_of_residence': 11, 'per:alternate_names': 12, 'per:other_family': 13, 'per:colleagues': 14, 'per:origin': 15, 'per:siblings': 16, 'per:spouse': 17, 'org:founded': 18, 'org:political/religious_affiliation': 19, 'org:member_of': 20, 'per:parents': 21, 'org:dissolved': 22, 'per:schools_attended': 23, 'per:date_of_death': 24, 'per:date_of_birth': 25, 'per:place_of_birth': 26, 'per:place_of_death': 27, 'org:founded_by': 28, 'per:religion': 29}
        label_num_arr = []

        for i in range(df_entity.shape[0]):
            label_num_arr.append(label_info[df_entity['label'].iloc[i]])
        
        df_entity['label_num'] = label_num_arr


    return df_entity

