''' 4. TF-IDF, Padding and GloVe Embedding '''

from preproc import *
from utilities import nfeature_accuracy_checker, dummy_fun, plot_acc_checker, pad
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer

process_num='Process 4/6 (TF-IDF, Padding and GloVe Embedding)'
print(process_num, 'started.')

print('TF-IDF vectorization ...')

''' TF-IDF '''

# ACCURACY CHECK WRT N OF FEATURES EXTRACTED
'''
cvec = CountVectorizer(lowercase=False)
tvec = TfidfVectorizer(lowercase=False)
lr = LogisticRegression()
n_features = np.arange(100,1001,100)
stop_words = stopwords.words('english')
# nfeature_accuracy_checker(X_train, y_train, X_val, y_val, vectorizer=cvec, n_features=n_features, stop_words=stop_words, ngram_range=(1, 1), classifier=lr)


feature_result_ugt = nfeature_accuracy_checker(X_train, y_train, X_val, y_val, vectorizer=tvec)                     # unigrams
feature_result_bgt = nfeature_accuracy_checker(X_train, y_train, X_val, y_val, vectorizer=tvec, ngram_range=(1, 2)) # bigrams
feature_result_tgt = nfeature_accuracy_checker(X_train, y_train, X_val, y_val, vectorizer=tvec, ngram_range=(1, 3)) # trigrams

nfeatures_plot_tgt = pd.DataFrame(feature_result_tgt,columns=['nfeatures','validation_accuracy','train_test_time'])
nfeatures_plot_bgt = pd.DataFrame(feature_result_bgt,columns=['nfeatures','validation_accuracy','train_test_time'])
nfeatures_plot_ugt = pd.DataFrame(feature_result_ugt,columns=['nfeatures','validation_accuracy','train_test_time'])

plot_acc_checker(nfeatures_plot_tgt, nfeatures_plot_bgt, nfeatures_plot_ugt)
'''

''' VECTORIZER FIT AND TRANSFORM '''

vect = TfidfVectorizer(stop_words=None, ngram_range=(1, 1), tokenizer=dummy_fun,
    preprocessor=dummy_fun, min_df=2, max_features=100).fit(X_train.text)

vect_transformed_X_train = vect.transform(X_train.text)
vect_transformed_X_val = vect.transform(X_val.text)
vect_transformed_X_test = vect.transform(X_test)

'''
feature_names = vect.get_feature_names()
dense = vect_transformed_X_train.todense() #returns data in matrix form
df = pd.DataFrame(dense, columns=feature_names)
df
#print(vect.idf_)
'''

''' PADDING '''
print('Padding ...')

word_tokenizer = Tokenizer()
word_tokenizer.fit_on_texts(X_train.text)

vocab_length = len(word_tokenizer.word_index) + 1
#print('Length of vocabulary:', vocab_length)
longest_train = X_train.text.str.len().max()
#print(longest_train)

train_padded_sentences = pad(X_train.text,word_tokenizer, longest_train)
val_padded_sentences = pad(X_val.text, word_tokenizer, longest_train)
test_padded_sentences = pad(X_test, word_tokenizer, longest_train)
#print(val_padded_sentences)
#print(vect_transformed_X_train.todense())

''' GLOVE EMBEDDING '''
# Mapping of the Glove Embedding dictionary to the training vocabulary
# https://www.kaggle.com/datasets/jdpaletto/glove-global-vectors-for-word-representation?resource=download

print('GloVe embedding ...')
embeddings_dictionary = dict()
embedding_dim = 100  # mapping each word in our dictionary with a 100-dimensional vector

# Load GloVe 100D embeddings
with open('../glove/glove.twitter.27B.100d.txt', encoding="utf8") as fp:
    for line in fp.readlines():
        records = line.split()
        word = records[0]
        vector_dimensions = np.asarray(records[1:], dtype='float32') # coefs
        embeddings_dictionary [word] = vector_dimensions
    fp.close()

# Now we will load embedding vectors of those words that appear in the
# Glove dictionary. Others will be initialized to 0.

embedding_matrix = np.zeros((vocab_length, embedding_dim))

for word, index in word_tokenizer.word_index.items():
    if index >= vocab_length:
        continue
    else:
        embedding_vector = embeddings_dictionary.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector

#print(embedding_matrix) # --> embedding_matrix.shape[0] == vocab_lenght

print(process_num, 'successfully finished.')