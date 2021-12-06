# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 12:10:28 2021

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

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier


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


# BOOSTING ENSEMBLE

# def nn_model():                                           
#     # create model
#     model = Sequential()
#     model.add(Dense(units=1000, input_dim=x_train.shape[1], kernel_initializer='normal', activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(units=1, activation='sigmoid'))
#     # Compile model
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#     return model

# nn_estimator = KerasClassifier(build_fn=nn_model, epochs=1000, verbose=0)


class customMLPClassifer(MLPClassifier):
    def resample_with_replacement(self, X_train, y_train, sample_weight):

        # normalize sample_weights if not already
        sample_weight = sample_weight / sample_weight.sum(dtype=np.float64)

        X_train_resampled = np.zeros((len(X_train), len(X_train[0])), dtype=np.float32)
        y_train_resampled = np.zeros((len(y_train)), dtype=np.int)
        for i in range(len(X_train)):
            # draw a number from 0 to len(X_train)-1
            draw = np.random.choice(np.arange(len(X_train)), p=sample_weight)

            # place the X and y at the drawn number into the resampled X and y
            X_train_resampled[i] = X_train[draw]
            y_train_resampled[i] = y_train[draw]

        return X_train_resampled, y_train_resampled


    def fit(self, X, y, sample_weight=None):
        if sample_weight is not None:
            X, y = self.resample_with_replacement(X, y, sample_weight)

        return self._fit(X, y, incremental=(self.warm_start and
                                            hasattr(self, "classes_")))


# Turn text dataframes into numpy arrays
X_train_text = X_train_text.to_numpy()
X_test_text = X_test_text.to_numpy()
X_val_text = X_val_text.to_numpy()


# BOOSTING ENSEMBLE - COMBINED DATA
# ADABOOST
num_estimators = 25

AdaBoost = AdaBoostClassifier(base_estimator= customMLPClassifer(),
                              n_estimators=num_estimators,
                              learning_rate=1)
AdaBoost.fit(X_train,y_train)

boostprediction = AdaBoost.score(X_train,y_train)

#Predict the response for test dataset
y_pred_test = AdaBoost.predict(X_test)
y_pred_val = AdaBoost.predict(X_val)




# BOOSTING ENSEMBLE - NOT COMBINED DATA

boost_NN_ratios = AdaBoostClassifier(base_estimator= customMLPClassifer(),
                                     n_estimators=num_estimators,
                                     learning_rate=1)
boost_NN_ratios.fit(X_train_ratio, y_train_ratio)

boost_NN_text = AdaBoostClassifier(base_estimator= customMLPClassifer(),
                                     n_estimators=num_estimators,
                                     learning_rate=1)
boost_NN_text.fit(X_train_text, y_train_ratio)

# EVALUATION 
print('\nADABOOST TRAINING ACCURACY combined is: ',boostprediction*100,'%')
print("ADABOOST TEST ACCURACY combined is:",accuracy_score(y_test, y_pred_test)*100,'%')
print("ADABOOST VAL ACCURACY combined is:",accuracy_score(y_val, y_pred_val)*100,'%')

print('\nADABOOST WITH ratios TRAINING SCORE:', boost_NN_ratios.score(X_train_ratio, y_train_ratio)*100,'%')
print('ADABOOST WITH ratios TEST SCORE:', boost_NN_ratios.score(X_test_ratio, y_test_ratio)*100,'%')
print('ADABOOST WITH ratios VAL SCORE:', boost_NN_ratios.score(X_val_ratio, y_val_ratio)*100,'%')
print('\nADABOOST WITH text TRAINING SCORE:', boost_NN_text.score(X_train_text, y_train_ratio)*100,'%')
print('ADABOOST WITH text TEST SCORE:', boost_NN_text.score(X_test_text, y_test_ratio)*100,'%')
print('ADABOOST WITH text VAL SCORE:', boost_NN_text.score(X_val_text, y_val_ratio)*100,'%')


predictions_ratios = []
for predictions in boost_NN_ratios.predict_proba(X_val_ratio):
    predictions_ratios.append(predictions[1])
# print(predictions_ratios)

predictions_text = []
for predictions in boost_NN_text.predict_proba(X_val_text):
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
print('\nACCURACY SCORE BOOSTING MODEL not combined IS:', accuracy_score(actual, predict)*100,'%')
print('\nCONFUSION MATRIX:\n', confusion_matrix(actual, predict))

print('predictions:', predict)

# %%
