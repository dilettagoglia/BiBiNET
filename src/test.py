''' 6. Final test '''
from data_prep import test_list
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utilities import wordcl

process_num='Process 6/6 (Final test prediction)'
print(process_num, 'started.')

def pred_test(model, test, test_padded_sentences, xai=True):
    '''

    :param model:
    :param test:
    :param test_padded_sentences:
    :param xai:
    :return:
    '''
    print('Final test set predictions... ')

    preds_test = model.predict(test_padded_sentences)
    y_test_pred = np.where(preds_test > 0.5, 1, 0)
    test['result']=y_test_pred

    orig_test = pd.concat([dataset for dataset in test_list])
    orig_test.reset_index(drop=True, inplace=True)
    # print(orig_test)
    orig_test['idx'] = orig_test.index
    test['idx'] = test.index
    orig_test.sort_values(by='idx')
    test_merged = test.merge(orig_test, on='idx', how='outer')

    print('Sample of texts in test set classified as hateful: \n')
    print(test_merged[test_merged.result==1][40150:40200])

    if xai==True:

        print('EXPLAINABILITY (visual evaluation) ')
        wordcl()


print(process_num, 'successfully finished.')