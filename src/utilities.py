''' AUXILIARY FUNCTIONS '''

import pandas as pd
import os
import re
import numpy as np
import seaborn as sns
from sklearn.utils import resample
import nltk
from nltk.corpus import stopwords
#nltk.download('stopwords')
from sklearn.metrics import classification_report, accuracy_score
from time import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from keras_preprocessing.sequence import pad_sequences
import itertools
import matplotlib.gridspec as gridspec
from mlxtend.plotting import plot_decision_regions
from sklearn.decomposition import PCA
from spellchecker import SpellChecker
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from wordcloud import WordCloud

pd.options.mode.chained_assignment = None  # default='warn'

################################################
#        2. Data preparation Functions
################################################

def twitter_preproc(dataset):
    '''
    This function performs the pre-processing of the Twitter data.
    We decided to consider both offensive and hateful tweets into 'hate' category of text.

    :param dataset: Twitter data
    :return: Processed dataset
    '''
    dataset['hate'] = 0
    dataset.loc[dataset['class'] < 2, "hate"] = 1 # merge offensive language and hate speech
    dataset.rename(columns={"tweet": "text"}, inplace=True)
    return dataset[['text', 'hate']]

def forum_preproc(dataset, dataset_2):
    '''
    This function performs the pre-processing of the Stormfront data.
    Textual data and the respective annotation are merged together.
    We considered 'relation' and 'skip' classes as not containing hate speech (see pdf report for more details).

    :param dataset: Forum text data
    :param dataset_2: Forum annotation data
    :return: Processed dataset
    '''
    # merge text splitted by comma into one column
    first_col = dataset.pop('id')
    dataset.insert(0, 'id', first_col)
    dataset['text'] = dataset[dataset.columns[1:]].apply(lambda x: ','.join(x.dropna().astype(str)), axis=1)
    dataset = dataset[['id','text']]
    dataset['id'] = dataset['id'].str.replace(".txt", "")
    result = pd.merge(dataset, dataset_2, on="id")
    result.rename(columns={"label": "hate"}, inplace=True)
    result = result.replace({"idk/skip":"noHate", "relation":"noHate"})
    result.hate = result.hate.map({'noHate':0 , 'hate':1})
    result.hate = result.hate.astype(int)
    return result[['id', 'text', 'hate']]

def wiki_preproc(dataset):
    '''
    This function performs the pre-processing of the Wikipedia data.
    Six classes of toxic comments are binarized in one variable, to indicate just the presence
    or absence of hate speech (see pdf report for more details).

    :param dataset: Wikipedia data
    :return: Processed dataset
    '''
    # binarization of six different hate labels into one variable
    dataset['hate'] = dataset.iloc[:,-6:].sum(axis=1)
    dataset.loc[dataset.hate > 0, "hate"] = 1
    return dataset[['id','text','hate']]

def train_test_spl(dataset):
    '''
    This function performs a preliminary split of the data into development set (95%) and test set (5%),
    for those dataset which were not already subdivided.

    :param dataset: data to be splitted
    :return: Development and test sets
    '''
    dataset_test = dataset.sample(frac=0.05) # test
    dataset_test.drop(['hate'], axis=1, inplace=True) # drop label from test set
    dataset_train = dataset[~dataset.isin(dataset_test)].dropna() # train
    return dataset_train, dataset_test

def substring_after(s, delim): # todo: delete
    return s.partition(delim)[2]

def csv_file(dirpath, output):
    '''
    This function puts all together the txt files from the folder that contains all the forum posts into a CSV.
    Each file contains a sentence.

    :param dirpath: directory path
    :param output: csv file
    :return: pandas dataframe
    '''
    csvout_lst = []
    files = [os.path.join(dirpath, fname) for fname in os.listdir(dirpath)]

    for filename in sorted(files):
        forum_data = pd.read_csv(filename, index_col=0, header=None, engine='python', quoting=3)
        forum_data['id'] = substring_after(str(filename), "\\")
        csvout_lst.append(forum_data)

    pd.concat(csvout_lst).to_csv(output)

def create_dataset(ds_list):
    '''
    This function merge all the training and test sets from different sources.

    :param ds_list: list of datasets to merge
    :return: unique dataset (DEV or TS)
    '''
    res = pd.concat([dataset for dataset in ds_list])
    res.drop("id", axis=1, inplace=True)  # drop id column
    res.reset_index(
        drop=True,  # to avoid the old index being added as a column
        inplace=False)
    return res

################################################
#        3. Pre-processing Functions
################################################

def clean_text(text):
    '''
    This function performs a complete cleaning of the corpus.
    N.B. some actions have been removed because they will be performed by of TweetTokenizer.

    :param text: corpus
    :return: cleaned corpus
    '''

    ## Convert words to Lower case and split them
    text = text.lower().split()

    ## Remove stop words (like 'the', 'to', 'you', ...)
    # sw=stopwords.words("english")
    # dataset["text"] = dataset["text"].apply(lambda x: " ".join(x for x in x.split() if x not in sw))
    stops = set(stopwords.words('english'))
    text = [w for w in text if not w in stops]  # and len(w) >= 3
    text = " ".join(text)

    ## clean the text
    # delete all characters, hashtags, re-tweet’s, links, except letters and numbers

    # discarded because of nltk TwitterTokenizer
    # text=re.sub(r"(@[A-Za-z0–9_]+)|([^-9A-Za-z \t])|(\w+:\/\/\S+)", '', text) # removing emojis and users IDs
    # text=re.sub(r'#', ' ', text) # hashtag

    text = re.sub(r'RT[\s]+', '', text)  # rt
    text = re.sub(r'https?:\/\/\S+', '', text)  # link
    text = re.sub(r"\b[a-zA-Z]\b", "", text)
    text = re.sub("\d", "", text)  # remove numbers
    text = re.sub("[^A-Za-z]", " ", text)
    text = re.sub(r'\s+', ' ', text)  # extra spaces

    return text

def correct_spellings(text):
    '''
    This function corrects the spelling of a given corpus passed as parameter, by using the SpellChecker Python module.

    See also:
    [pyspellchecker 0.6.3](https://pypi.org/project/pyspellchecker/)

    :param text: corpus
    :return: corpus corrected
    '''

    spell = SpellChecker()
    corrected_text = []
    misspelled_words = spell.unknown(text.split())
    for word in text:
        if word in misspelled_words:
            corrected_text.append(spell.correction(word))
        else:
            corrected_text.append(word)
    return " ".join(corrected_text)

def tokenize_func(text, tokenizer):
    '''
    This function performs the tokenization of the text, by using the tokenizer passed as parameter.
    In this project we use the TweetTokenizer from NLTK module.

    See also:
    [NLTK TwitterTokenizer documentation](https://www.nltk.org/api/nltk.tokenize.casual.html)

    :param text: corpus to be tokenized
    :param tokenizer: tokenizer
    :return: tokenized corpus
    '''

    tokens = tokenizer.tokenize(text)
    #tokens = [token.strip() for token in tokens]
    #filtered_tokens = [token for token in tokens if token not in stopword_list]
    return tokens

def lemmatize_func(text, lemmatizer):
    '''
    This function performs the lemmatization of the text, by using the lemmatizer passed as parameter.
    In this project we use the WordNetLemmatizer from NLTK module.

    :param text: corpus to lemmatize
    :param lemmatizer: lemmatizer
    :return: lemmatized corpus
    '''

    lem = lemmatizer.lemmatize(text)
    return lem

def eda(train_list, train):
    '''
    This function performs an Exploratory Data Analysis (EDA) on corpora.
    It analyses and plots textual properties and checks the class balance.

    :param train_list: list of TR sets form different sources
    :param train: DEV set
    '''

    print('EDA (Exploratory Data Analysis of Text Data) ...')
    for ds in train_list:
        ds.reset_index(drop=True, inplace=True)
        ds['length'] = ds['text'].apply(len)
        ds['count_'] = ds.groupby('length')['length'].transform('count')

    plt.style.use('ggplot')
    f, ax = plt.subplots(4, 1, figsize=(24, 8))

    # Get a color map
    my_cmap = cm.get_cmap('jet')
    my_norm = Normalize(vmin=0, vmax=8)

    no_hate = []
    hate = []
    plts = no_hate, hate
    axes_ = []

    for i in range(4):
        no_hate.append(train_list[i][train_list[i].hate == 0])  # 4 dataframes
        hate.append(train_list[i][train_list[i].hate == 1])  # 4 dataframes
    for p in plts:
        for i in range(4):
            axes_.append(p[i].length.sort_values(ascending=False).unique())
            axes_.append(p[i].groupby(['length'])['count_'].mean())

    ax[0].bar(axes_[0], axes_[1], color='g')
    ax[1].bar(axes_[2], axes_[3], color='r')
    ax[0].set_title('Non-hateful')
    ax[1].set_title('Hateful')

    ax[2].hist(train[train.hate == 0].text.str.len(), color='g')
    ax[3].hist(train[train.hate == 1].text.str.len(), color='r')
    ax[2].set_title('Non-hateful')
    ax[3].set_title('Hateful')

    for i in range(0, 2):
        ax[i].set_ylabel('Count')
        ax[i].set_xlabel('Length (characters)')
    for i in range(2, 4):
        ax[i].set_ylabel('Count')
        ax[i].set_xlabel('Length (tokens)')
    plt.suptitle('Length of Posts/Tweets/Comments in TR set')
    plt.show()
    plt.savefig('../img/text_len.png')

    print('Checking class balance ...')

    # dealing with unbalanced classes
    # visualize with column chart and pie chart to examine the distribution of hate speech and non-hate speech data in the dataset.

    f, ax = plt.subplots(1, 2, figsize=(24, 8))
    train['hate'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)
    ax[0].set_title('Distribution')
    ax[0].set_ylabel('')
    sns.countplot('hate', data=train, ax=ax[1])
    ax[1].set_title('hate')
    plt.show()
    plt.savefig('../img/orig_class_distrib.png')

def tr_upsample(tr, plt_=True):
    '''
    This function performs the resampling of the training set in order to fix class imbalance.
    Minority class (hate, label 1) is upsampled.

    :param tr: imbalanced TR set
    :param plt: (boolean) whether generating plot of class distribution (comparison before/after resampling)
    :return: upsampled TR set
    '''
    train_majority = tr[tr.hate == 0]
    train_minority = tr[tr.hate == 1]
    train_minority_upsampled = resample(train_minority,
                                        replace=True,
                                        n_samples=len(train_majority),
                                        random_state=123)
    train_upsampled = pd.concat([train_minority_upsampled, train_majority])

    if plt_==True:

        plt.figure(figsize=(8, 6))
        sns.set_style('darkgrid')
        sns.histplot(data=tr['hate'], color='black', legend=True)
        sns.histplot(data=train_upsampled['hate'], color='orange', legend=True)
        plt.legend(['Initial_Data', 'Resampled_Data'])
        plt.show()
        plt.savefig('../img/bef_aft_resampling.png')

    return train_upsampled

################################################
#      4. Text transformation Functions
################################################

def dummy_fun(doc):
    '''
    This function just returns the parameter. Needed to break the embedded tokenizer of TF-IDF
    since tokenization already happened with NLTK TweetTokenizer.

    :param doc: text
    :return: text
    '''
    return doc

def accuracy_summary(pipeline, x_train, y_train, x_test, y_test):
    '''
    This function computes accuracy and time for vectorizer fit on data, and evaluates the prediction
    with a logistic regression model. It is used to check the best accuracy results for vectorizer
    extracting unigrams, bigrams and trigrams.

    :param pipeline: vectorizer (CountVectorizer or TfidfVectorizer) + classifier (LogisticRegression)
    :param x_train: TR set
    :param y_train: TR labels
    :param x_test: VAL set
    :param y_test: VAL labels
    :return: accuracy ad time
    '''

    if len(x_test[y_test == 0]) / (len(x_test)*1.) > 0.5:
        null_accuracy = len(x_test[y_test == 0]) / (len(x_test)*1.)
    else:
        null_accuracy = 1. - (len(x_test[y_test == 0]) / (len(x_test)*1.))
    t0 = time()
    sentiment_fit = pipeline.fit(x_train.text.apply(str), y_train)
    y_pred = sentiment_fit.predict(x_test.text.apply(str))
    train_test_time = time() - t0
    accuracy = accuracy_score(y_test, y_pred)
    print("null accuracy: {0:.2f}%".format(null_accuracy*100))
    print("accuracy score: {0:.2f}%".format(accuracy*100))
    if accuracy > null_accuracy:
        print("model is {0:.2f}% more accurate than null accuracy".format((accuracy-null_accuracy)*100))
    elif accuracy == null_accuracy:
        print("model has the same accuracy with the null accuracy")
    else:
        print("model is {0:.2f}% less accurate than null accuracy".format((null_accuracy-accuracy)*100))
    print("train and test time: {0:.2f}s".format(train_test_time))
    print("-"*80)
    return accuracy, train_test_time

def nfeature_accuracy_checker(X_train, y_train, X_val, y_val,
                              vectorizer=CountVectorizer(lowercase=False), n_features=np.arange(1000,10001,1000),
                              stop_words=stopwords.words('english'), classifier=LogisticRegression(), ngram_range=(1,1)):
    '''
    This function builts a Pipeline vectorizer + classifier to be fit and evaluated by the 'accuracy_summary' funtion.

    :param X_train: TR set
    :param y_train: TR labels
    :param X_val: VAL set
    :param y_val: VAL labels
    :param vectorizer: vectorizer, default CountVectorizer
    :param n_features: range of features to extract
    :param stop_words: list containing stop words, default NLTK english stopwords
    :param classifier: model for performance evaluation, default LogisticRegression
    :param ngram_range: lower and upper boundary of the range of n-values for different n-grams to be extracted, default unigrams
    :return: features extracted, accuracy and time
    '''
    result = []
    #print (classifier)
    print ("\n")
    for n in n_features:
        vectorizer.set_params(stop_words=stop_words, max_features=n, ngram_range=ngram_range)
        checker_pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', classifier)
        ])
        print("Validation result for {} features".format(n))
        nfeature_accuracy,tt_time = accuracy_summary(checker_pipeline, X_train, y_train, X_val, y_val)
        result.append((n,nfeature_accuracy,tt_time))
    return result

def plot_acc_checker(nfeatures_plot_tgt, nfeatures_plot_bgt, nfeatures_plot_ugt):
    '''
    This function plots the results obtained from 'nfeature_accuracy_checker' function.

    :param nfeatures_plot_tgt: extracted features using trigrams
    :param nfeatures_plot_bgt: extracted features using bigrams
    :param nfeatures_plot_ugt: extracted features using unigrams
    '''
    plt.figure(figsize=(8, 6))
    plt.plot(nfeatures_plot_tgt.nfeatures, nfeatures_plot_tgt.validation_accuracy, label='trigram tfidf vectorizer',
             color='royalblue')
    # plt.plot(nfeatures_plot_tg.nfeatures, nfeatures_plot_tg.validation_accuracy,label='trigram count vectorizer',linestyle=':', color='royalblue')
    plt.plot(nfeatures_plot_bgt.nfeatures, nfeatures_plot_bgt.validation_accuracy, label='bigram tfidf vectorizer',
             color='orangered')
    # plt.plot(nfeatures_plot_bg.nfeatures, nfeatures_plot_bg.validation_accuracy,label='bigram count vectorizer',linestyle=':',color='orangered')
    plt.plot(nfeatures_plot_ugt.nfeatures, nfeatures_plot_ugt.validation_accuracy, label='unigram tfidf vectorizer',
             color='gold')
    # plt.plot(nfeatures_plot_ug.nfeatures, nfeatures_plot_ug.validation_accuracy, label='unigram count vectorizer',linestyle=':',color='gold')
    plt.title("N-gram(1~3) test result : Accuracy")
    plt.xlabel("Number of features")
    plt.ylabel("Validation set accuracy")
    plt.legend()
    plt.savefig('.../img/tfidf_accuracy.png')

def embed(corpus, word_tokenizer):
    '''
    Transforms each record (text) in corpus to a sequence of integers.

    See also:
    [tf.keras.preprocessing.text.Tokenizer](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer)

    :param corpus: textual data (list of strings)
    :param word_tokenizer: TensorFlow text tokenization utility class.
    :return: transformed text in numerical (integer) data
    '''

    return word_tokenizer.texts_to_sequences(corpus)

def pad(corpus, word_tokenizer, longest_train):
    '''
    This function transforms a list (of length num_samples) of sequences (lists of integers) into a
    2D Numpy array of shape (num_samples, num_timesteps).
    num_timesteps is either the maxlen argument if provided, or the length of the longest sequence in the list.

    See also:
    [tf.keras.utils.pad_sequences](https://www.tensorflow.org/api_docs/python/tf/keras/utils/pad_sequences)

    :param corpus: text previously transformed in numerical form by the 'embed' function
    :param word_tokenizer: Text tokenization utility class.
    :param longest_train: longest sentence in TR set
    :return: 2-dim numpy array
    '''

    return pad_sequences(
        embed(corpus,word_tokenizer),
        longest_train,
        padding='post')

################################################
#           0. Model Functions
################################################

def report_scores(test_label, test_pred):
    '''
    This function prints the evaluation metrics for the predicted labels.

    :param test_label: true labels
    :param test_pred: predicted labels
    '''
    print(classification_report(test_label,
                            test_pred,
                            target_names=['0', '1']))

def plot_model_curves(model, out_file_name):
    '''
    This function plots the Loss and Accuracy curves (for both training and validation)
    related to the model passed as parameter, and save the image into the corresponding folder.

    :param model: classifier
    :param out_file_name: name of the image to be saved
    '''

    acc = model.history.history['accuracy']
    val_acc = model.history.history['val_accuracy']
    loss = model.history.history['loss']
    val_loss = model.history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(epochs, acc, label='Training', c='r')
    ax1.plot(epochs, val_acc, label='Validation', c='b')
    ax1.set_title('Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax2.plot(epochs, loss, label='Training', c='r')
    ax2.plot(epochs, val_loss, label='Validation', c='b')
    ax2.set_title('Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')

    ax1.legend()
    ax2.legend()
    plt.show()
    plt.savefig(f'../img/{str(out_file_name)}.png')

def plot_c_matrix(test_set, test_label, test_pred, classifier, classifier_name, axs=None):
    '''
    This function computes and plots the confusion matrix related to classifier predictions.

    :param test_set: VAL set
    :param test_label: VAL labels
    :param test_pred: predicted val labels
    :param classifier: model
    :param classifier_name: (string) name of the model
    :param axs: (boolean) needed for matplotlib
    '''

    cm = confusion_matrix(test_label, test_pred)
    if axs:
        plot_confusion_matrix(classifier, test_set, test_label, ax=axs)
    else:
        plot_confusion_matrix(classifier, test_set, test_label)
    plt.title('Confusion matrix of ' + classifier_name)
    plt.show()

def plot_decision_boundary(train_set, train_label, classifier, classifier_name, axs=None):
    '''
    This function computes and plots the decision boundary of the classifier passed as parameter,
    using Principal Component Analysis (PCA).

    See also:
    [sklearn.decomposition.PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)

    :param train_set: TR set
    :param train_label: TR labels
    :param classifier: model
    :param classifier_name: (string) name of the model
    :param axs: (boolean) needed for matplotlib
    '''

    X = train_set
    y = np.array(train_label)

    pca = PCA(n_components=2)
    X = pca.fit_transform(X)
    gs = gridspec.GridSpec(2, 2)
    fig = plt.figure(figsize=(20, 10))

    labels = [classifier_name]
    for clf, lab, grd in zip([classifier],
                             labels,
                             itertools.product([0, 1], repeat=2)):

        clf.fit(X, y)
        ax = plt.subplot(gs[grd[0], grd[1]])
        if axs:
            fig = plot_decision_regions(X=X, y=y, clf=clf, legend=2, ax=axs)
        else:
            fig = plot_decision_regions(X=X, y=y, clf=clf, legend=2)
        plt.title('Decision boundary of ' + lab)

def save_model(model, model_name, del_model=False):
    '''
    This function exports and saves the trained model into an external HDF5 file.

    :param model: model to export
    :param model_name: (string) name to assign to the file
    :param del_model: (boolean) wheter to delete the existing model from memory, default=False
    '''

    print('Saving Bi-LSTM model ...')
    model.save(f'../model/{str(model_name)}.h5')  # creates a HDF5 file 'model.h5'
    # For only storing the model definition, obtain its description as a JSON or YAML:
    # json_string = model.to_json()
    # yaml_string = model.to_yaml()
    if del_model==True:
        del model  # deletes the existing model

def predict_on_val(model, val):
    '''
    This function performs the prediction of the model on validation set and binarizes the results.

    :param model: model that predicts
    :param val: validation set
    :return: predicted (binarized) values
    '''

    print('Model prediction on validation set ...')
    preds = model.predict(val)
    # print(preds)
    y_pred = np.where(preds > 0.5, 1, 0)
    print('Predicted labels for validation set\n', y_pred)
    return y_pred

def write_model_summary(model, y_val, y_pred):
    '''
    This function writes the model summary and performance on an external txt file.

    :param model: desired model
    :param y_val: true validation labels
    :param y_pred: predicted labels
    '''

    print('Writing output in external file ...')
    f1 = open("lstm_models.txt", "a")

    print('\nBi-LSTM model\n')
    print(model.summary(), file=f1)
    print('\n Model performance on validation set\n', report_scores(y_val, y_pred))
    f1.close()

def wordcl(test):
    '''
    This function performs the visual evaluation and explanation of the final test predictions by plotting
    the wordcloud of both classes.

    :param test: TS set
    '''

    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    text_pos = []
    text_neg = []
    for words in test['text'][test.result == 0]:
        text_pos.append(" ".join(words))
    for words in test['text'][test.result == 1]:
        text_neg.append(" ".join(words))
    train_cloud_pos = WordCloud(collocations=False, background_color='white').generate(''.join(text_pos))
    train_cloud_neg = WordCloud(collocations=False, background_color='black').generate(''.join(text_neg))
    axs[0].imshow(train_cloud_pos, interpolation='bilinear')
    axs[0].axis('off')
    axs[0].set_title('Non-Hate Text')
    axs[1].imshow(train_cloud_neg, interpolation='bilinear')
    axs[1].axis('off')
    axs[1].set_title('Hate Text')
    plt.show()
    plt.savefig('../img/wordcloud_2.png')