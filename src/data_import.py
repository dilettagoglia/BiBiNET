''' 1. Data Import '''

import pandas as pd

process_num='Process 1/6 (Data Import)'
print(process_num, 'started. \nImporting data ...')

# read csv file
twitter_data_1 = pd.read_csv('../data/twitter_1/twitter_dataset.csv', index_col=[0], sep=",", low_memory=False)

twitter_2_train = pd.read_csv('../data/twitter_2/train.csv', sep=",", low_memory=False)
twitter_2_train.rename(columns={"label": "hate", "tweet":"text"}, inplace=True)

twitter_2_test = pd.read_csv('../data/twitter_2/test.csv', sep=",", low_memory=False)
twitter_2_test.rename(columns={"tweet":"text"}, inplace=True)

forum_data_annot = pd.read_csv('../data/forum_data/annotations_metadata.csv', sep=",", low_memory=False)
forum_data_annot.rename(columns={"file_id": "id"}, inplace=True)

wikipedia_data_train = pd.read_csv('../data/wikipedia_data/train.csv', low_memory=False)
wikipedia_data_train.rename(columns={"comment_text": "text"}, inplace=True)

wikipedia_test = pd.read_csv('../data/wikipedia_data/test.csv', low_memory=False)
wikipedia_test.rename(columns={"comment_text": "text"}, inplace=True)

print(process_num, 'successfully finished.')