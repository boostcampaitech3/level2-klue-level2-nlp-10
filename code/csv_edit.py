import pandas as pd
def wordlen(text):
    return len(text) < 400

def edit_func1(df):
    sub_type = []
    sub_word = []
    sub_start = []
    sub_end = []
    obj_type = []
    obj_word = []
    obj_start = []
    obj_end = []
    id = []

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

    df_entity2=df_entity[df_entity['sentence'].apply(wordlen)]
    df_entity2=df_entity2.drop_duplicates(['subj_word','sentence','obj_word','subj_start','obj_start','label'],keep='first')
    df_entity3 = df_entity2.drop([6749,8364,22258,277,25094])

    return df_entity3

def edit_func2(df):
    sub_type = []
    sub_word = []
    sub_start = []
    sub_end = []
    obj_type = []
    obj_word = []
    obj_start = []
    obj_end = []
    id = []

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

    return df_entity
