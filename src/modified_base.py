"""
beating the benchmark @StumbleUpon Evergreen Challenge
__author__ : Abhishek Thakur
"""

# -*- coding: utf-8 -*-
import numpy as np
from sklearn import metrics,preprocessing,cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.linear_model as lm
import pandas as p
from scipy.sparse import csr_matrix, coo_matrix, vstack, hstack

def main():

  print "loading data.."
  traindata_url = list(np.array(p.read_table('../data/train.tsv'))[:,0])
  testdata_url = list(np.array(p.read_table('../data/test.tsv'))[:,0])
  
  traindata_content = list(np.array(p.read_table('../data/train.tsv'))[:,2])
  testdata_content = list(np.array(p.read_table('../data/test.tsv'))[:,2])

  traindata = [traindata_url[i]+" "+traindata_content[i]  for i in xrange(len(traindata_content))]
  testdata = [testdata_url[i]+" "+testdata_content[i]  for i in xrange(len(testdata_content))]


  y = np.array(p.read_table('../data/train.tsv'))[:,-1]

  tfv = TfidfVectorizer(min_df=3,  max_features=None, strip_accents='unicode',  
        analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1)

  rd = lm.LogisticRegression(penalty='l2', dual=True, tol=0.0001, 
                             C=1, fit_intercept=True, intercept_scaling=1.0, 
                             class_weight=None, random_state=None)

#  X_all_url = traindata_url + testdata_url
#  print "fitting url pipeline"
#  tfv.fit(X_all_url)
#  print "transforming url data"
#  X_all_url = tfv.transform(X_all_url)
#
#  X_all_content = traindata_content + testdata_content
#  print "fitting content pipeline"
#  tfv.fit(X_all_content)
#  print "transforming content data"
#  X_all_content = tfv.transform(X_all_content)

  X_all_combine = traindata + testdata
  print "fitting combine pipeline"
  tfv.fit(X_all_combine)
  print "transforming combine data"
  X_all_combine = tfv.transform(X_all_combine)

  lentrain = len(traindata_url)

#  X_all = csr_matrix(hstack([X_all_url,X_all_content,X_all_combine])); 
  X_all = csr_matrix(hstack([X_all_combine])); 

  X = X_all[:lentrain]
  X_test = X_all[lentrain:]

  print "20 Fold CV Score: ", np.mean(cross_validation.cross_val_score(rd, X, y, cv=20, scoring='roc_auc'))

  print "training on full data"
  rd.fit(X,y)
  pred = rd.predict_proba(X_test)[:,1]
  print "pred len:" + str(len(pred))
  testfile = p.read_csv('../data/test.tsv', sep="\t", na_values=['?'], index_col=1)
  pred_df = p.DataFrame(pred, index=testfile.index, columns=['label'])
  pred_df.to_csv('benchmark.csv')
  print "submission file created.."

  f = open("predict_res","w+");
  for pred_item in pred:
    f.write(str(pred_item)+'\n');
  f.close();

if __name__=="__main__":
  main()

