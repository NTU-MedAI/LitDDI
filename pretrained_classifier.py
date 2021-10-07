import pandas as pd
import pickle
import joblib


def get_abstract_vector(vector, vector_list):
    vector_list2 = [vector[k] for k in vector_list ]
    vector_df = pd.DataFrame(vector_list2)
    cut_word_vector = vector_df.mean(axis=0).values
    return cut_word_vector


def predict(input_v):
    pre_result = clf_2.predict(input_v)
    if pre_result[0] == 1:
        class_4 = clf_4.predict(input_v)
        print('True:', class_4[0])
    else:
        print('False')


disease = '乳腺癌'
embeddings = pickle.load(open(disease+'DDI_embedding(20210916).pkl', 'rb'))
key_list = pickle.load(open(disease+'DDI_Cui(20210916).pkl', 'rb'))

clf_2 = joblib.load(disease+'2class.m')
clf_4 = joblib.load(disease+'4class.m')

CHEM1 = input('CHEM1:')
CHEM2 = input('CHEM2:')
cui_pair = [key_list.index(CHEM1), key_list.index(CHEM2)]
input_embedding = get_abstract_vector(embeddings, cui_pair)
demo1 = pd.DataFrame([input_embedding])
predict(demo1)

