#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 16:44:44 2019

@author: liujiachen
"""
FILE_NAME_OUT="MBTI_tfidf"
QUERY=1000;
k=100;

import pandas as pd
import numpy as np

import re
from stemming.porter2 import stem
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


print("Starting generate "+FILE_NAME_OUT)
data = pd.read_csv('mbti_1.csv')
types = np.unique(np.array(data['type']))
labelCoder = LabelEncoder()
inty = labelCoder.fit_transform(data['type'])
data['posts']=[re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', sentence) for sentence in data['posts']]
data['posts']=[re.sub("[^a-zA-Z]"," ",sentence) for sentence in data['posts']]
data['posts']=[re.sub(' +',' ',sentence).lower() for sentence in data['posts']]
data['posts']=[[stem(word) for word in sentence.split(" ")] for sentence in data['posts']]
data['posts']=[" ".join(sentence) for sentence in data['posts']]
vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=1600, max_df=0.5)
''', max_features=1500, max_df=0.5'''
X = vectorizer.fit_transform(data['posts'])
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(X)
base=tfidf.todense()
DIM=base.shape[1];
NUM=base.shape[0];

print("Extraction featuer ready")

#np.savetxt(FILE_NAME_OUT+'.txt', base, fmt='%8f')


a=np.arange(base.shape[0])
a=a.reshape((base.shape[0],1))
base = np.hstack((a,base))
#np.savetxt(FILE_NAME_OUT+'_base.txt', base, fmt='%8f')

print("Base data ready")

learn=base[int(round(base.shape[0]*9/10)):-1] 
#np.savetxt(FILE_NAME_OUT+'_learn.txt', learn, fmt='%8f')


print("Learn data ready")


query=base[0:QUERY] 
#np.savetxt(FILE_NAME_OUT+'_query.txt', query, fmt='%8f')

print("Query data ready")


gt = np.zeros( (QUERY,k) ) # k=100

for i in range(QUERY):
   
  dis_list=np.zeros((1,NUM))
  
  for j in range(NUM):
      dis=0
      
      for d in range(DIM):
           
          dis+= pow( query[i,d+1]-base[j,1+d],2)
          
      #print(dis)  
      dis_list[0,j]=dis
  rank_no=np.argsort(dis_list)
  gt[i]=rank_no[0][1:k+1]
 # if i%round(QUERY/10) ==0 :
  if i%10 ==0 :
      print("Find ",i, "th gt")
  
a=np.arange(QUERY)
a=a.reshape((QUERY,1))
gt = np.hstack((a,gt))


np.savetxt(FILE_NAME_OUT+'_gt.txt', gt , fmt='%d')










