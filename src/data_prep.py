''' 2. Data Preparation '''

from data_import import *
from utilities import twitter_preproc, forum_preproc, wiki_preproc, train_test_spl, csv_file, create_dataset

process_num='Process 2/6 (Data Preparation)'
print(process_num, 'started.')
print('Splitting TR/TS ...')

twitter_1 = twitter_preproc(twitter_data_1)
twitter_1_train, twitter_1_test = train_test_spl(twitter_1)

'''In the repository of the forum data each sample of text extracted is stored as a single text file.
In order to limit the code and solving the task of appending all texts together in a single csv file this 
simple function csv_file is provided.'''
#dirpath_f = 'data/forum_data/all_files'
output_f = 'data/forum_data/all_files.csv'
#csv_file(dirpath_f, output_f)
forum_data = pd.read_csv('../data/forum_data/all_files.csv', low_memory=False)
forum = forum_preproc(forum_data, forum_data_annot)
forum_train, forum_test = train_test_spl(forum)

wikipedia_train = wiki_preproc(wikipedia_data_train)

#check
if (len(twitter_1_test) + len(twitter_1_train) != len(twitter_1)) | (len(forum_train) + len(forum_test) != len(forum)):
    raise ValueError('Mismatching length of data.')
else:

    print('Creating dataset ...')

    train_list = [twitter_1_train, twitter_2_train, forum_train, wikipedia_train]
    test_list = [twitter_1_test, twitter_2_test, forum_test, wikipedia_test]

    train = create_dataset(train_list)
    test = create_dataset(test_list)

    len_train, len_test = 0, 0
    for i in train_list: len_train+=len(i)
    for i in test_list: len_test+=len(i)

if (len_train != len(train)) | (len_test != len(test)):
    raise ValueError('Mismatching length of data.')
else:
    print(process_num, 'successfully finished.')
    print('Dataset contains', len(train), 'TR records and', len(test), 'TS records.')

