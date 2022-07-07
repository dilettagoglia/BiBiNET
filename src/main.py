''' 0. MAIN '''

import matplotlib.pyplot as plt
from transform import *
from utilities import plot_model_curves, plot_c_matrix, plot_decision_boundary, report_scores, save_model, predict_on_val, write_model_summary
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from keras.models import load_model
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.metrics import classification_report, make_scorer, f1_score, accuracy_score, confusion_matrix
from IPython.display import Image
from keras.utils.vis_utils import model_to_dot
import scikitplot as skplt
from sklearn.preprocessing import MinMaxScaler
from classifiers import *
from test import *


print('Building models for preliminary experiments (on TF-IDF only)...')

''' preliminary experiments on simple models with only TD-IDF vector '''

''' Multinomial Nayve Bayes '''
mnb = MultinomialNB()
train_pred_mnb, val_pred_mnb, pred_proba_mnb = simple_classifier(
    mnb, vect_transformed_X_train, y_train, vect_transformed_X_val, pred_probabilities=True
)

''' Logistic Regression '''
modelLR = LogisticRegression()
predictionsLR_train, predictionsLR = simple_classifier(
    modelLR, vect_transformed_X_train, y_train, vect_transformed_X_val
)
#print(sum(predictionsLR==1),len(y_val),f1_score(y_val,predictionsLR))

''' Random Forest '''
modelRF = RandomForestClassifier()
predictionsRF_train, predictionsRF = simple_classifier(
    modelRF, vect_transformed_X_train, y_train, vect_transformed_X_val
)
#print(sum(predictionsRF==1),len(y_val),f1_score(y_val,predictionsRF))

''' Multilayer Perceptron '''
'''
mlp = MLPClassifier()
predictionsMLP_train, predictionsMLP = simple_classifier(
    mlp, vect_transformed_X_train, y_train, vect_transformed_X_val
)
'''

print('Plotting ROC curves, confusion matrices and decision boundaries for each classifier ...')

''' ROC curves '''

skplt.metrics.plot_roc(y_val, pred_proba_mnb)                   # mnb
plt.suptitle('Multinomial Naive Bayes')
#plt.savefig('img/mnb_roc.png')

pred_proba_lr = modelLR.predict_proba(vect_transformed_X_val)   # lr
skplt.metrics.plot_roc(y_val, pred_proba_lr)
plt.suptitle('Logistic Regression')
#plt.savefig('img/lr_roc.png')

pred_proba_rf = modelRF.predict_proba(vect_transformed_X_val)   # rf
skplt.metrics.plot_roc(y_val, pred_proba_rf)
plt.suptitle('Random Forest')
#plt.savefig('img/rf_roc.png')

''' Confusion matrix '''

plot_c_matrix(vect_transformed_X_val, y_val, val_pred_mnb, mnb, 'Multinomial Naive Bayes')  # mnb
#plt.savefig('img/mnb_conf_matr.png')

plot_c_matrix(vect_transformed_X_val, y_val, predictionsLR, modelLR, 'Logistic Regression') # lr
#plt.savefig('img/lr_conf_matr.png')

plot_c_matrix(vect_transformed_X_val, y_val, predictionsRF, modelRF, 'Random Forest')       # rf
#plt.savefig('img/rf_conf_matr.png')

''' Decision boundaries '''

scaler = MinMaxScaler(feature_range=(0, 1))
x_mnb_scaled=scaler.fit_transform(vect_transformed_X_train.toarray()) 
#plot_decision_boundary(x_mnb_scaled, y_train, mnb, 'Multinomial Naive Bayes')'''                   # mnb

plot_decision_boundary(vect_transformed_X_train.toarray(), y_train, modelLR, 'Logistic Regression') # lr
#plt.savefig('img/lr_dec_bound.png')

plot_decision_boundary(vect_transformed_X_train.toarray(), y_train, modelRF, 'Random Forest')       # rf
#plt.savefig('img/rf_dec_bound.png')


print('Model evaluation on both TR and VAL sets ...')
# Output file
f = open("models_scores.txt", "a")
print ('MNB performance on train set: \n', report_scores(y_train, train_pred_mnb), file=f)
print ('MNB performance on val set: \n', report_scores(y_val, val_pred_mnb), file=f)
print ('LR performance on train set: \n', report_scores(y_train, predictionsLR_train), file=f)
print ('LR performance on val set: \n', report_scores(y_val, predictionsLR), file=f)
print('RF performance on train set: \n', report_scores(y_train, predictionsRF_train), file=f)
print('RF performance on val set: \n', report_scores(y_val, predictionsRF), file=f)
f.close()



''' GATED RECURRENT NN MODELS (Bi-LSTM) '''

''' MODEL 1 (Keras Sequential API) '''
print('Building gated RNN (model #1) ...')
neurons = 8
model = glove_lstm(embedding_matrix, longest_train, neurons)

# Train the model
# history = rnn_train(train_padded_sentences, y_train, val_padded_sentences, y_val, early_stop=False, 20, 256)

# Save and Evaluate the model
'''
model_name='model'
save_model(model, model_name, del_model=True)
model_stored = load_model(f'model/{str(model_name)}.h5') # returns a compiled model identical to the previous one

print('Evaluating model and plotting curves ...')
# Loss and accuracy curves 
plot_model_curves(model_stored, 'lstm_loss_curve_model_1')

# MODEL EVALUATION ON VAL SET 
y_pred = predict_on_val(model_stored, val_padded_sentences)
'''


''' MODEL 2 (Keras Functional API) '''

print('Building gated RNN (model #2) ...')

sequence_input = Input(shape=(longest_train,))
embed_size = embedding_matrix.shape[1]
x = Embedding(embedding_matrix.shape[0], embed_size, weights=[embedding_matrix], trainable=False)(sequence_input)
x = Bidirectional(LSTM(neurons, return_sequences=False, dropout=0.6, recurrent_dropout=0.4))(x)
# x = GlobalMaxPool1D()(x)
x = Dense(neurons, activation="relu")(x)
x = Dropout(0.6)(x)
dense_input = Input(shape=(embed_size,))  # 100
dense_vector = BatchNormalization()(dense_input)
feature_vector = concatenate([x, dense_vector])
feature_vector = Dense(neurons // 2, activation="relu")(feature_vector)
feature_vector = Dropout(0.6)(feature_vector)
output = Dense(1, activation="sigmoid")(feature_vector)

model_2 = Model(inputs=[sequence_input, dense_input], outputs=output)
model_2.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
#print(model_2.summary())

# Train the model
'''
history = rnn_train([train_padded_sentences, vect_transformed_X_train.toarray()],
                    y_train, [val_padded_sentences,vect_transformed_X_val.toarray()], y_val,
                    10, 256)
'''

# Save and Evaluate the model
model_name='model'
# save_model(model_2, model_name, del_model=True)
model_stored = load_model(f'model/{str(model_name)}.h5') # returns a compiled model identical to the previous one

'''
Image(model_to_dot(model_stored, show_shapes=True, show_dtype=True, expand_nested=True,
                  show_layer_activations=True).create(prog='dot', format='png'))'''

print('Evaluating model and plotting curves ...')
''' Loss and accuracy curves '''
plot_model_curves(model_stored, 'lstm_loss_curve_model_2')

''' MODEL EVALUATION ON VAL SET '''
y_pred = predict_on_val(model_stored, val_padded_sentences) # todo sostituire con [val_padded_sentences,vect_transformed_X_val.toarray()]

''' EXTERNAL OUTPUT SUMMARY '''
write_model_summary(model_2, y_val, y_pred)

''' FINAL TEST SET PREDICTIONS '''
pred_test(model, test, test_padded_sentences, xai=True)


