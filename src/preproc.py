''' 3. Pre-processing '''

from data_prep import train, test, train_list
from utilities import clean_text, tokenize_func, lemmatize_func, eda, tr_upsample
# nltk.download('stopwords')
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

process_num='Process 3/6 (Pre-processing)'
print(process_num, 'started.')
print('Preprocessing (cleaning, tokenizing and lemmatizing text)...')

''' NLP PREPROCESSING '''

train["text"] = train["text"].apply(clean_text)
test["text"] = test["text"].apply(clean_text)

dataset = pd.concat([test, train], axis=0)

''' TOKENIZATION, STEMMING, LEMMATIZATION '''

#stopword_list = set(stopwords.words('english'))
tokenizer = TweetTokenizer()
lemmatizer = WordNetLemmatizer()
stemmer = SnowballStemmer("english")

train.text = train.apply(lambda row: tokenize_func(row['text'], tokenizer), axis=1)
train.text = train.text.apply(lambda row: [lemmatize_func(word, lemmatizer) for word in row])
#train.text = train.text.apply(lambda row: [stemmer.stem(word) for word in row])
test.text = test.apply(lambda row: tokenize_func(row['text'],tokenizer), axis=1)
test.text = test.text.apply(lambda row: [lemmatize_func(word, lemmatizer) for word in row])
#train.text = train.apply(lambda row: correct_spellings(row['text']), axis=1)
#test.text = test.apply(lambda row: correct_spellings(row['text']), axis=1)

# print(train)
# print(test)

''' PRELIMINARY ANALYSIS '''
eda(train_list, train)

''' SPLIT '''

if len(dataset[dataset.hate.isna()]) != len(test):
    raise ValueError('Test set length is not correct.')

else:
    print('Splitting dataset on TR/VAL/TS ...')

    ''' TEST SET '''
    test.reset_index(drop=True, inplace=True)
    X_test, y_test = test['text'], dataset[dataset.hate.isna()]['hate']

    ''' TR AND VAL PROPORTIONS '''
    msk = np.random.rand(len(train)) < 0.9
    train_dataset = train[msk]
    val = train[~msk]

    print('Resampling and shuffling TR set ...')

    ''' RESAMPLING (only on TR set) '''
    train_upsampled = tr_upsample(train_dataset)
    print('Upsampled TR set classes: \n', train_upsampled['hate'].value_counts())

    plt.figure(figsize=(8,6))
    sns.set_style('darkgrid')
    sns.histplot(data = train['hate'], color='black', legend=True)
    sns.histplot(data = train_upsampled['hate'], color = 'orange', legend=True)
    plt.legend(['Initial_Data', 'Resampled_Data'])
    plt.show()
    plt.savefig('img/bef_aft_resampling.png')

    ''' RANDOM PERMUTATIONS '''
    train_upsampled.reset_index(drop=True, inplace=True)
    train_upsampled.reindex(np.random.permutation(train_upsampled.index))

    print(train_upsampled)
    print(val)

    ''' TRAINING AND VALIDATION SETS '''
    X_train, X_val, y_train, y_val = train_upsampled.drop(
        columns = ['hate']).copy(), val.drop(
        columns = ['hate']).copy(), train_upsampled['hate'], val['hate']

    print(process_num, 'successfully finished.')

    print('\n Obtained dataset split: ')
    print('------ TR ------')
    print(X_train.shape, y_train.shape)
    print('------ VAL ------')
    print(X_val.shape, y_val.shape)
    print(' ------ TS ------')
    print(X_test.shape, y_test.shape)


print(process_num, 'successfully finished.')