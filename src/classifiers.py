''' 5. Models '''
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers import *

process_num='Process 5/6 (Models)'
print(process_num, 'started.')

def simple_classifier(model, X_train, y_train, X_val, pred_probabilities=False):
    '''

    :param model: classifier
    :param X_train: TR set vectorized with TF-IDF
    :param y_train: TR labels
    :param X_val: VAL set vectorized with TF-IDF
    :param pred_probabilities: (boolean) add probability prediction on VAL set, default=None
    :return: prediction on TR, prediction on VAL, (predicted probabilities on VAL)
    '''

    # FIT
    model.fit(X_train, y_train)

    # PREDICT
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    # PREDICT PROBABILITIES
    if pred_probabilities==True:
        pred_proba = model.predict_proba(X_val)
        # print(pred_proba)
        return train_pred, val_pred, pred_proba
    else:
        return train_pred, val_pred

def glove_lstm(embedding_matrix, longest_train, neurons):
    '''
    Keras Sequential API
    :return: Bi-LSTM model
    '''
    mod = Sequential()

    mod.add(Embedding(
        input_dim=embedding_matrix.shape[0],  # == vocab_len
        output_dim=embedding_matrix.shape[1],  # 100
        weights=[embedding_matrix],
        input_length=longest_train,
        trainable=False
    ))

    mod.add(Bidirectional(LSTM(
        # longest_train,
        neurons,
        return_sequences=False  # ???
        # recurrent_dropout=0.2
    )))

    # mod.add(GlobalMaxPool1D())
    # mod.add(BatchNormalization())
    mod.add(Dropout(0.8))
    mod.add(Dense(neurons // 2, activation="relu"))
    # mod.add(Dropout(0.8))
    # mod.add(Dense(neurons//3, activation = "relu"))
    # mod.add(Dropout(0.8))
    mod.add(Dense(1, activation='sigmoid'))
    mod.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # ignore_class_accuracy(0)
    # new metric defined: ignore_class_accuracy(0) added For handling the bad effect of padding, since The padding influences the accuracy

    return mod

def rnn_train(model, X_train, y_train, X_val, y_val, epochs, batch_size, early_stop=True):
    '''
    This function perform the fit of the model
    :param model: Bi-LSTM
    :param X_train: TR set padded
    :param y_train: TR labels
    :param X_val: VAL set padded
    :param y_val: VAL labels
    :param epochs: epochs
    :param batch_size: batch size
    :param early_stop: (bolean) if to implement early stopping techniques during learning (boolean=True)
    :return: history
    '''
    print('Training the model ...')

    checkpoint = ModelCheckpoint(
        'model.h5',
        monitor='val_loss',
        verbose=1,
        save_best_only=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        verbose=1,
        patience=5,
        min_lr=0.001
    )

    if early_stop == True:
        early_stopping = EarlyStopping(monitor="val_accuracy", patience=2, verbose=1)
        callbacks = [reduce_lr, checkpoint, early_stopping]
    else:
        callbacks = [reduce_lr, checkpoint]

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        verbose=1,
        callbacks=callbacks
    )

    return history

print(process_num, 'successfully finished.')