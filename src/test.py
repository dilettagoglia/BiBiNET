''' 6. Final test '''
from data_prep import test_list
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

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

        fig, axs = plt.subplots(1,2 , figsize=(16,8))
        text_pos = []
        text_neg = []
        for words in test['text'][test.result == 0]:
            text_pos.append(" ".join(words))
        for words in test['text'][test.result == 1]:
            text_neg.append(" ".join(words))
        train_cloud_pos = WordCloud(collocations = False, background_color = 'white').generate(''.join(text_pos))
        train_cloud_neg = WordCloud(collocations = False, background_color = 'black').generate(''.join(text_neg))
        axs[0].imshow(train_cloud_pos, interpolation='bilinear')
        axs[0].axis('off')
        axs[0].set_title('Non-Hate Text')
        axs[1].imshow(train_cloud_neg, interpolation='bilinear')
        axs[1].axis('off')
        axs[1].set_title('Hate Text')
        plt.show()
        plt.savefig('img/wordcloud_2.png')


    print(process_num, 'successfully finished.')

print(process_num, 'successfully finished.')