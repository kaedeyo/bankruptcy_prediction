# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 11:31:58 2021

@author: onjad
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import os

from janome.tokenizer import Tokenizer
from janome.analyzer import Analyzer
from janome.charfilter import *
from janome.tokenfilter import *
import nltk
import re

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix,accuracy_score, roc_curve, auc

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier

import statistics

# TEXTUAL DATA IMPORT
# TRAINING SET
#Bankrupt_files
bankrupted_risk_reports = os.listdir('C:/Users/onjad/OneDrive/Documents/RESEARCH PHD/article 4/CODE/DATAFRAMES/risk_info/bankrupted/trainB')
bankrupted_risk_reports = [file for file in bankrupted_risk_reports]

bankrupted_file = []
for report in range(0, len(bankrupted_risk_reports)):
    with open(f"C:/Users/onjad/OneDrive/Documents/RESEARCH PHD/article 4/CODE/DATAFRAMES/risk_info/bankrupted/trainB/{bankrupted_risk_reports[report]}", encoding="utf8") as file:
        filesB = file.read()
        bankrupted_file.append(filesB)

df_bankrupted = pd.DataFrame(bankrupted_file)
df_bankrupted.columns = ['text_file']
df_bankrupted['Y'] = 0 #Y=0 means bankrupted
# print(df_bankrupted)

#Nonbankrupt_files
nonbankrupted_risk_reports = os.listdir('C:/Users/onjad/OneDrive/Documents/RESEARCH PHD/article 4/CODE/DATAFRAMES/risk_info/nonbankrupted/trainNB')
nonbankrupted_risk_reports = [file for file in nonbankrupted_risk_reports]

nonbankrupted_file = []
for report in range(0, len(nonbankrupted_risk_reports)):
    with open(f"C:/Users/onjad/OneDrive/Documents/RESEARCH PHD/article 4/CODE/DATAFRAMES/risk_info/nonbankrupted/trainNB/{nonbankrupted_risk_reports[report]}", encoding="utf8") as file:
        filesNB = file.read()
        nonbankrupted_file.append(filesNB)

df_nonbankrupted = pd.DataFrame(nonbankrupted_file)
df_nonbankrupted.columns = ['text_file']
df_nonbankrupted['Y'] = 1 #Y=1 means non-bankrupted
# print(df_nonbankrupted)

concatenated_train = [df_bankrupted, df_nonbankrupted]
df_train = pd.concat(concatenated_train, ignore_index= True)
# print(df_train)

# VALIDATION SET
#Bankrupt_files
bankrupted_risk_reports_val = os.listdir('C:/Users/onjad/OneDrive/Documents/RESEARCH PHD/article 4/CODE/DATAFRAMES/risk_info/bankrupted/valB')
bankrupted_risk_reports_val = [file for file in bankrupted_risk_reports_val]

bankrupted_file_val = []
for report in range(0, len(bankrupted_risk_reports_val)):
    with open(f"C:/Users/onjad/OneDrive/Documents/RESEARCH PHD/article 4/CODE/DATAFRAMES/risk_info/bankrupted/valB/{bankrupted_risk_reports_val[report]}", encoding="utf8") as file:
        filesB_val = file.read()
        bankrupted_file_val.append(filesB_val)

df_bankrupted_val = pd.DataFrame(bankrupted_file_val)
df_bankrupted_val.columns = ['text_file']
df_bankrupted_val['Y'] = 0 #Y=0 means bankrupted
# print(df_bankrupted_val)

#Nonbankrupt_files
nonbankrupted_risk_reports_val = os.listdir('C:/Users/onjad/OneDrive/Documents/RESEARCH PHD/article 4/CODE/DATAFRAMES/risk_info/nonbankrupted/valNB')
nonbankrupted_risk_reports_val = [file for file in nonbankrupted_risk_reports_val]

nonbankrupted_file_val = []
for report in range(0, len(nonbankrupted_risk_reports_val)):
    with open(f"C:/Users/onjad/OneDrive/Documents/RESEARCH PHD/article 4/CODE/DATAFRAMES/risk_info/nonbankrupted/valNB/{nonbankrupted_risk_reports_val[report]}", encoding="utf8") as file:
        filesNB_val = file.read()
        nonbankrupted_file_val.append(filesNB_val)

df_nonbankrupted_val = pd.DataFrame(nonbankrupted_file_val)
df_nonbankrupted_val.columns = ['text_file']
df_nonbankrupted_val['Y'] = 1 #Y=1 means non-bankrupted
# print(df_nonbankrupted_val)

concatenated_val = [df_bankrupted_val, df_nonbankrupted_val]
df_val = pd.concat(concatenated_val, ignore_index=True)
# print(df_val)

# TEST SET

#Bankrupt_files (test data)
testbankrupted_risk_reports = os.listdir('C:/Users/onjad/OneDrive/Documents/RESEARCH PHD/article 4/CODE/DATAFRAMES/risk_info/bankrupted/testB')

testbankrupted_file = []
for report in range(0, len(testbankrupted_risk_reports)):
    with open(f"C:/Users/onjad/OneDrive/Documents/RESEARCH PHD/article 4/CODE/DATAFRAMES/risk_info/bankrupted/testB/{testbankrupted_risk_reports[report]}", encoding="utf8") as file:
        testfilesB = file.read()
        testbankrupted_file.append(testfilesB)

df_bankrupted_test = pd.DataFrame(testbankrupted_file)
df_bankrupted_test.columns = ['text_file']
df_bankrupted_test['Y'] = 0

#Nonbankrupt_files (test data)
testnonbankrupted_risk_reports = os.listdir('C:/Users/onjad/OneDrive/Documents/RESEARCH PHD/article 4/CODE/DATAFRAMES/risk_info/nonbankrupted/testNB')

testnonbankrupted_file = []
for report in range(0, len(testnonbankrupted_risk_reports)):
    with open(f"C:/Users/onjad/OneDrive/Documents/RESEARCH PHD/article 4/CODE/DATAFRAMES/risk_info/nonbankrupted/testNB/{testnonbankrupted_risk_reports[report]}", encoding="utf8") as file:
        testfilesNB = file.read()
        testnonbankrupted_file.append(testfilesNB)

df_nonbankrupted_test = pd.DataFrame(testnonbankrupted_file)
df_nonbankrupted_test.columns = ['text_file']
df_nonbankrupted_test['Y'] = 1

concatenated_test = [df_bankrupted_test, df_nonbankrupted_test]
df_test = pd.concat(concatenated_test, ignore_index=True)
# df_test


# RATIO DATA IMPORT
# TRAINING SET

traindata = pd.read_csv('C:/Users/onjad/OneDrive/Documents/RESEARCH PHD/article 4/CODE/DATAFRAMES/ratios/train_ratio.csv')
# print('TRAINING DATASET \n:', traindata.describe().transpose())
X_train_ratio = traindata.drop('Y', axis=1)
y_train_ratio = traindata['Y']
# print(sns.countplot(x = 'Y', data=traindata))

# VALIDATION SET

valdata = pd.read_csv('C:/Users/onjad/OneDrive/Documents/RESEARCH PHD/article 4/CODE/DATAFRAMES/ratios/val_ratio.csv')
# print('VALIDATION DATASET \n:', valdata.describe().transpose())
X_val_ratio = valdata.drop('Y', axis=1)
y_val_ratio = valdata['Y']

# TEST SET

testdata = pd.read_csv('C:/Users/onjad/OneDrive/Documents/RESEARCH PHD/article 4/CODE/DATAFRAMES/ratios/test_ratio.csv')
# print('TEST DATASET \n:', testdata.describe().transpose())
# print('TEST DATASET \n:', testdata)
X_test_ratio = testdata.drop('Y', axis=1)
y_test_ratio = testdata['Y']

# COMBINE BOTH DATA INTO ONE TABLE
# ADD THE A NEW COLUMN FOR TEXT FILE

X_train_ratio['text'] = df_train['text_file']
X_val_ratio['text'] = df_val['text_file']
X_test_ratio['text'] = df_test.drop('Y', axis=1)

# CREATE NEW DATAFRAME FOR COMBINED DATA

X_train_both = X_train_ratio
X_val_both = X_val_ratio
X_test_both = X_test_ratio

y_train_both = y_train_ratio
y_val_both = y_val_ratio
y_test_both = y_test_ratio

print('Length of X (train): {} | Length of Y (train): {}'.format(len(X_train_both), len(y_train_both)))
print('Length of X (validation): {} | Length of Y (validation): {}'.format(len(X_val_both), len(y_val_both)))
print('Length of X (test): {} | Length of Y (test): {}'.format(len(X_test_both), len(y_test_both)))


# TEXT PREPROCESSING
def clean_data(risk_report_test):
    """Returns cleaned text data."""
    
    with open(f"C:/Users/onjad/OneDrive/Documents/RESEARCH PHD/article 4/CODE/stop_words.txt", mode="r", encoding="utf8") as file:
        stop_words = file.read()
    
    with open(f"C:/Users/onjad/OneDrive/Documents/RESEARCH PHD/article 4/CODE/stop_words_accounting.txt", mode="r", encoding="utf8") as file:
        stop_words_acc = file.read()
    
    with open(f"C:/Users/onjad/OneDrive/Documents/RESEARCH PHD/article 4/CODE/onaji_risk_vocab.txt", mode="r", encoding="utf8") as file:
        onaji_risk_vocab = file.read()
        
    
    char_filters = [UnicodeNormalizeCharFilter(), 
                    RegexReplaceCharFilter(u'キャッシュ・フロー', u'キャッシュフロー')]
    tokenizer = Tokenizer()
    token_filters = [CompoundNounFilter(), 
                     POSStopFilter(['記号','助詞', '数', '数接続', '接続詞','連体詞', '接頭詞', '名詞,数','非自立', '代名詞', '自動詞', '他動詞']), 
                     LowerCaseFilter()]


    a = Analyzer(char_filters=char_filters, tokenizer=tokenizer, token_filters=token_filters)
    
    def filter(text):
        """
        :param text: str
        :rtype : str
        """
       # アルファベットと半角英数と記号と改行とタブを排除
        text = re.sub(r'[a-zA-Z0-9¥"¥.¥,¥@]+', '', text)
        text = re.sub(r'[!"“#$%&()\*\+\-\.,\/:;<=>?@\[\\\]^_`{|}~]', '', text)
        text = re.sub(r'[\n|\r|\t]', '', text)
        
        # 日本語以外の文字を排除
        jp_chartype_tokenizer = nltk.RegexpTokenizer(u'([ぁ-んー]+|[ァ-ンー]+|[\u4e00-\u9FFF]+|[ぁ-んァ-ンー\u4e00-\u9FFF]+)')
        text = ''.join(jp_chartype_tokenizer.tokenize(text))
        
        return text

    nosymbol_test = filter(risk_report_test)
    
    doc = []
    for token in a.analyze(nosymbol_test):
        doc.append(token.surface)
    doc = [x for x in doc if x not in stop_words]
    doc = [x for x in doc if x not in stop_words_acc]
    doc = [x for x in doc if x not in onaji_risk_vocab]
    doc = np.array(doc)
    
    return doc


df_text_only = pd.concat([X_train_both['text'], X_val_both['text'], X_test_both['text']], ignore_index=True)
# print(df_text_only)


clean_file_transformer = CountVectorizer(analyzer=clean_data).fit(df_text_only)
# print(clean_file_transformer.get_feature_names())
print(len(clean_file_transformer.vocabulary_))


clean_file_bow = clean_file_transformer.transform(df_text_only)
# print(clean_file_bow)
print('Shape of Sparse Matrix: ', clean_file_bow.shape)


transformer_tfidf = TfidfTransformer().fit(clean_file_bow)
clean_file_tfidf = transformer_tfidf.transform(clean_file_bow)
# print(clean_file_tfidf)
print(clean_file_tfidf.shape)


# RATIO PREPROCESSING
scaler = MinMaxScaler()
X_train_ratio = scaler.fit_transform(X_train_ratio.drop('text', axis=1))
X_val_ratio = scaler.transform(X_val_ratio.drop('text', axis=1))
X_test_ratio = scaler.transform(X_test_ratio.drop('text', axis=1))


# DATA SPLIT
# RATIOS
Xtrain_ratio = pd.DataFrame(X_train_ratio)
Xval_ratio = pd.DataFrame(X_val_ratio)
Xtest_ratio = pd.DataFrame(X_test_ratio)

# rename columns
Xtrain_ratio.columns = ['X19','X20','X29','X47', 'X51', 'X54', 'X59', 'X62','X63','X64','X67', 'X69', 'X70', 'X71','X72', 'X73', 'X74', 'X76','X77', 'X79', 'X80', 'X84']
Xval_ratio.columns = ['X19','X20','X29','X47', 'X51', 'X54', 'X59', 'X62','X63','X64','X67', 'X69', 'X70', 'X71','X72', 'X73', 'X74', 'X76','X77', 'X79', 'X80', 'X84']
Xtest_ratio.columns = ['X19','X20','X29','X47', 'X51', 'X54', 'X59', 'X62','X63','X64','X67', 'X69', 'X70', 'X71','X72', 'X73', 'X74', 'X76','X77', 'X79', 'X80', 'X84']

print(Xtrain_ratio.shape, Xval_ratio.shape, Xtest_ratio.shape)


# TEXT
X = clean_file_tfidf
X = pd.DataFrame.sparse.from_spmatrix(X)
X_train_text = X[:92]
X_val_text = pd.DataFrame(X[92:112])
X_test_text = pd.DataFrame(X[112:])
X_val_text.index.delete
X_val_text = X_val_text.reset_index(drop=True)
X_test_text.index.delete
X_test_text = X_test_text.reset_index(drop=True)
print(X_train_text.shape, X_val_text.shape, X_test_text.shape)


# INPUT VALUES X

X_train = pd.concat([Xtrain_ratio, X_train_text], ignore_index=True, axis=1)
X_val = pd.concat([Xval_ratio, X_val_text], ignore_index=True, axis=1)
X_test = pd.concat([Xtest_ratio, X_test_text], ignore_index=True, axis=1)

print(X_train.shape, X_val.shape, X_test.shape)

# TARGET VALUE Y

y_train = pd.DataFrame(y_train_ratio)
y_val = pd.DataFrame(y_val_ratio)
y_test = pd.DataFrame(y_test_ratio)

print(y_train.shape, y_val.shape, y_test.shape)


# BAGGING ENSEMBLE - COMBINED DATA
# EVALUATION ON TRAIN DATASET
bag_NN = BaggingClassifier(base_estimator=MLPClassifier(), 
                           n_estimators=20, 
                           max_samples=0.8, 
                           oob_score=True, 
                           random_state=0)
bag_NN.fit(X_train, y_train_ratio)



# 
# estimators = list(range(1, 25))
# accuracy = []

# for n_estimators in estimators:
#     clf = BaggingClassifier(base_estimator=MLPClassifier(),
#                             max_samples=0.8,
#                             n_estimators=n_estimators,
#                             oob_score=True, 
#                             random_state=0,)
#     clf.fit(X_train_text, y_train_ratio)
#     acc = clf.score(X_val_text, y_val_ratio)
#     accuracy.append(acc)
    

# plt.plot(estimators, accuracy)
# plt.xlabel("Number of estimators")
# plt.ylabel("Accuracy")
# plt.show()


# BAGGING ENSEMBLE - NOT COMBINED DATA
# num_estimators = 100

bag_NN_ratios = BaggingClassifier(base_estimator=MLPClassifier(), 
                                  n_estimators=12, 
                                  max_samples=0.8, oob_score=True, 
                                  random_state=0)
bag_NN_ratios.fit(X_train_ratio, y_train_ratio)

bag_NN_text = BaggingClassifier(base_estimator=MLPClassifier(), 
                                n_estimators=13, 
                                max_samples=0.8, oob_score=True, 
                                random_state=0)
bag_NN_text.fit(X_train_text, y_train_ratio)




predictions_ratios = []
for predictions in bag_NN_ratios.predict_proba(X_val_ratio):
    predictions_ratios.append(predictions[1])
# print(predictions_ratios)
predictions_text = []
for predictions in bag_NN_text.predict_proba(X_val_text):
    predictions_text.append(predictions[1])
# print(predictions_text)


average = [statistics.mean(k) for k in zip(predictions_ratios, predictions_text)]
print('average:', average)

pred_overall = []

for predictions in average:
    if predictions < 0.50:
        predictions = 0
    else:
        predictions=1
    pred_overall.append(predictions)
    
predict = np.array(pred_overall)
actual = np.array(y_val)


# %%
# EVALUATION ON TRAIN DATASET - combined
print('\nBAGGING TRAINING SCORE combined:',bag_NN.oob_score_*100,'%')
# EVALUATION ON TEST DATASET - combined
print('BAGGING TEST SCORE combined:',bag_NN.score(X_test, y_test)*100,'%')
print('BAGGING VAL SCORE combined:',bag_NN.score(X_val, y_val)*100,'%')


# EVALUATION ON TRAIN DATASET - not combined
print('\nBAGGING WITH ratios TRAINING SCORE:', bag_NN_ratios.oob_score_*100,'%')
print('BAGGING WITH text TRAINING SCORE:', bag_NN_text.oob_score_*100,'%')

# EVALUATION ON TEST DATASET - not combined
print('\nBAGGING WITH ratios TEST SCORE:', bag_NN_ratios.score(X_test_ratio, y_test_ratio)*100,'%')
print('BAGGING WITH text TEST SCORE:', bag_NN_text.score(X_test_text, y_test_ratio)*100,'%')

print('\nBAGGING WITH ratios VAL SCORE:', bag_NN_ratios.score(X_val_ratio, y_val_ratio)*100,'%')
print('BAGGING WITH text VAL SCORE:', bag_NN_text.score(X_val_text, y_val_ratio)*100,'%')

print('\nACCURACY SCORE BAGGING MODEL not combined IS:', accuracy_score(actual, predict)*100,'%')
print('\nCONFUSION MATRIX:\n', confusion_matrix(actual, predict))

print('predictions:', predict)

# %%

