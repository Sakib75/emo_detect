import os
import pandas as pd

def SUBESCO_label(name):
    return name.split('_')[5].lower()
def TESS_label(name):
    return name.split('_')[2].split('.')[0].lower()
    
def load_dataset(path,corpus,label_method):
    paths = []
    labels = []
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            label = label_method(name)
            labels.append(label)
            paths.append(os.path.join(root,name))
    print(f"{len(labels)} {corpus} DATA Loaded")
    df = pd.DataFrame()
    df['speech'] = paths
    df['label'] =labels
    df['corpus'] = corpus
    df['label'] = df['label'].replace('ps','surprise')
    return df

def get_gender(speech):
    gender = speech.split('\\')[-1].split('_')[0].lower()
    return gender
def merge_datasets(bangla,english_tess):
    all_datasets = []
    if(bangla):
        bangla_dataset = load_dataset("./SUBESCO/","Bangla",SUBESCO_label)
        bangla_dataset['gender'] = bangla_dataset['speech'].apply(get_gender)
        all_datasets.append(bangla_dataset)
    if(english_tess):
        english_dataset = load_dataset("./TESS/","English",TESS_label)
        all_datasets.append(english_dataset)
        

    final_dataset = pd.concat(all_datasets)
    final_dataset.reset_index(inplace=True, drop=True) 
    return final_dataset

print(merge_datasets(bangla=True, english_tess=False))







