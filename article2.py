# -*- coding: utf-8 -*-

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

#%%
#Bankrupt_files
bankrupted_risk_reports = os.listdir('C:/Users/onjad/OneDrive/Documents/RESEARCH PHD/Codes/risk/consolidated2/bankrupted/trainB')
bankrupted_risk_reports = [file for file in bankrupted_risk_reports]

bankrupted_file = []
for report in range(0, len(bankrupted_risk_reports)):
    with open(f"C:/Users/onjad/OneDrive/Documents/RESEARCH PHD/Codes/risk/consolidated2/bankrupted/trainB/{bankrupted_risk_reports[report]}", encoding="utf8") as file:
        filesB = file.read()
        bankrupted_file.append(filesB)

df_bankrupted = pd.DataFrame(bankrupted_file)
df_bankrupted.columns = ['text_file']
df_bankrupted['Y'] = 0 #Y=0 means bankrupted
# print(bankrupted_data)

#%%
#Nonbankrupt_files
nonbankrupted_risk_reports = os.listdir('C:/Users/onjad/OneDrive/Documents/RESEARCH PHD/Codes/risk/consolidated2/nonbankrupted/trainNB')
nonbankrupted_risk_reports = [file for file in nonbankrupted_risk_reports]

nonbankrupted_file = []
for report in range(0, len(nonbankrupted_risk_reports)):
    with open(f"C:/Users/onjad/OneDrive/Documents/RESEARCH PHD/Codes/risk/consolidated2/nonbankrupted/trainNB/{nonbankrupted_risk_reports[report]}", encoding="utf8") as file:
        filesNB = file.read()
        nonbankrupted_file.append(filesNB)

df_nonbankrupted = pd.DataFrame(nonbankrupted_file)
df_nonbankrupted.columns = ['text_file']
df_nonbankrupted['Y'] = 1 #Y=1 means non-bankrupted
# print(df_nonbankrupted)

#%%
#Bankrupt_files
bankrupted_risk_reports_val = os.listdir('C:/Users/onjad/OneDrive/Documents/RESEARCH PHD/Codes/risk/consolidated2/bankrupted/valB')
bankrupted_risk_reports_val = [file for file in bankrupted_risk_reports_val]

bankrupted_file_val = []
for report in range(0, len(bankrupted_risk_reports_val)):
    with open(f"C:/Users/onjad/OneDrive/Documents/RESEARCH PHD/Codes/risk/consolidated2/bankrupted/valB/{bankrupted_risk_reports_val[report]}", encoding="utf8") as file:
        filesB_val = file.read()
        bankrupted_file_val.append(filesB_val)

df_bankrupted_val = pd.DataFrame(bankrupted_file_val)
df_bankrupted_val.columns = ['text_file']
df_bankrupted_val['Y'] = 0 #Y=0 means bankrupted
# print(df_bankrupted_val)

#Nonbankrupt_files
nonbankrupted_risk_reports_val = os.listdir('C:/Users/onjad/OneDrive/Documents/RESEARCH PHD/Codes/risk/consolidated2/nonbankrupted/valNB')
nonbankrupted_risk_reports_val = [file for file in nonbankrupted_risk_reports_val]

nonbankrupted_file_val = []
for report in range(0, len(nonbankrupted_risk_reports_val)):
    with open(f"C:/Users/onjad/OneDrive/Documents/RESEARCH PHD/Codes/risk/consolidated2/nonbankrupted/valNB/{nonbankrupted_risk_reports_val[report]}", encoding="utf8") as file:
        filesNB_val = file.read()
        nonbankrupted_file_val.append(filesNB_val)

df_nonbankrupted_val = pd.DataFrame(nonbankrupted_file_val)
df_nonbankrupted_val.columns = ['text_file']
df_nonbankrupted_val['Y'] = 1 #Y=1 means non-bankrupted
# print(df_nonbankrupted_val)

concatenated_val = [df_bankrupted_val, df_nonbankrupted_val]
df_val = pd.concat(concatenated_val)
# print(df_val)

#%%
#Bankrupt_files (test data)
testbankrupted_risk_reports = os.listdir('C:/Users/onjad/OneDrive/Documents/RESEARCH PHD/Codes/risk/consolidated2/bankrupted/testB')

testbankrupted_file = []
for report in range(0, len(testbankrupted_risk_reports)):
    with open(f"C:/Users/onjad/OneDrive/Documents/RESEARCH PHD/Codes/risk/consolidated2/bankrupted/testB/{testbankrupted_risk_reports[report]}", encoding="utf8") as file:
        testfilesB = file.read()
        testbankrupted_file.append(testfilesB)

df_bankrupted_test = pd.DataFrame(testbankrupted_file)
df_bankrupted_test.columns = ['text_file']
df_bankrupted_test['Y'] = 0

#Nonbankrupt_files (test data)
testnonbankrupted_risk_reports = os.listdir('C:/Users/onjad/OneDrive/Documents/RESEARCH PHD/Codes/risk/consolidated2/nonbankrupted/testNB')

testnonbankrupted_file = []
for report in range(0, len(testnonbankrupted_risk_reports)):
    with open(f"C:/Users/onjad/OneDrive/Documents/RESEARCH PHD/Codes/risk/consolidated2/nonbankrupted/trainNB/{nonbankrupted_risk_reports[report]}", encoding="utf8") as file:
        testfilesNB = file.read()
        testnonbankrupted_file.append(testfilesNB)

df_nonbankrupted_test = pd.DataFrame(testnonbankrupted_file)
df_nonbankrupted_test.columns = ['text_file']
df_nonbankrupted_test['Y'] = 1

concatenated_test = [df_bankrupted_test, df_nonbankrupted_test]
df_test = pd.concat(concatenated_test)
# df_test

#%%
from sklearn.utils import shuffle

df_test = shuffle(df_test)
# df_test

#%%
concatenated = [df_bankrupted, df_nonbankrupted, df_val, df_test]
df = pd.concat(concatenated, ignore_index=True)
# print(df)
# print(df[92:112]) #val data (20 samples)
print(df[112:]) #test data (26 samples)

#%%
df['length'] = df['text_file'].apply(len)
df.head()

#%%
df.groupby('Y').describe()

#%%
df.hist(column='length', by='Y', bins=50, figsize=(16,8))

#%%

def clean_data(risk_report_test):
    """Returns cleaned text data."""
    
    with open(f"C:/Users/onjad/OneDrive/Documents/RESEARCH PHD/Codes/risk/stop_words.txt", mode="r", encoding="utf8") as file:
        stop_words = file.read()
    
    with open(f"C:/Users/onjad/OneDrive/Documents/RESEARCH PHD/Codes/risk/stop_words_accounting.txt", mode="r", encoding="utf8") as file:
        stop_words_acc = file.read()
    
    with open(f"C:/Users/onjad/OneDrive/Documents/RESEARCH PHD/Codes/risk/onaji_vocab.txt", mode="r", encoding="utf8") as file:
        onaji_vocab = file.read()
        
    
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
    doc = [x for x in doc if x not in onaji_vocab]
    doc = np.array(doc)
    
    return doc

#%%
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
clean_file_transformer = CountVectorizer(analyzer=clean_data).fit(df['text_file'])

#%%
# print(clean_file_transformer.get_feature_names())
print(len(clean_file_transformer.vocabulary_))

#%%
clean_file_bow = clean_file_transformer.transform(df['text_file'])
# print(clean_file_bow)
print('Shape of Sparse Matrix: ', clean_file_bow.shape)

#%%
transformer_tfidf = TfidfTransformer().fit(clean_file_bow)
clean_file_tfidf = transformer_tfidf.transform(clean_file_bow)

# print(clean_file_tfidf)
print(clean_file_tfidf.shape)

#%%
X = clean_file_tfidf
X = pd.DataFrame.sparse.from_spmatrix(X)
X_train = X[:92]
X_val = X[92:112]
X_test = X[112:]

y = np.array(df['Y'])
y_train = y[:92]
y_val = y[92:112]
y_test = y[112:]

print(X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape)

#%%
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential([
    Dense(units=256, activation='relu', input_dim=clean_file_tfidf.shape[1]),
    Dropout(0.5),
    Dense(units=1, activation='sigmoid')
])

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

#%%
# TRAIN MODEL

from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)

model.fit(X_train, y_train, validation_data=(X_val, y_val) , epochs=5000, verbose=2, callbacks=[early_stop])

#%%
# PLOT THE LOSS

losses = pd.DataFrame(model.history.history)
losses.plot()
plt.show()

#%%
# NN model

pred_NN = []

for predictions in model.predict(X_test):
    if predictions < 0.5:
        predictions = 0
    else:
        predictions = 1
    pred_NN.append(predictions)
    
pred = np.array(pred_NN)
true = np.array(y_test)

print("PREDICTED CLASS: ", pred)
print("ACTUAL CLASS: ", true, '\n')

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(classification_report(true, pred))
print('\n CONFUSION MATRIX:\n', confusion_matrix(true, pred))
print('\n\n ACCURACY SCORE IS:', accuracy_score(true, pred))

#%%
# Classifier (Naive Bayes)

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
X_train_nb = np.array(X[:112])
y_train_nb = np.array(y[:112])
nb.fit(X_train_nb,y_train_nb)

pred_naive = nb.predict(X_test)

print("PREDICTED CLASS: ", pred_naive)
print("ACTUAL CLASS: ", true)

print(classification_report(true, pred_naive))
print('\n CONFUSION MATRIX:\n', confusion_matrix(true, pred_naive))
print('\n\n ACCURACY SCORE IS:', accuracy_score(true, pred_naive), '\n')

#%%
# Classifier (SVM)

from sklearn.svm import SVC

svm_model = SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
X_train_svm = np.array(X[:112])
y_train_svm = np.array(y[:112])

svm_model.fit(X_train_svm, y_train_svm)

pred_svm = svm_model.predict(np.array(X_test))

print("PREDICTED CLASS: ", pred_svm)
print("ACTUAL CLASS: ", true)

print(classification_report(true, pred_svm))
print('\n CONFUSION MATRIX:\n', confusion_matrix(true, pred_svm))
print('\n\n ACCURACY SCORE IS:', accuracy_score(true, pred_svm))
