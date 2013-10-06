import cPickle
import gzip
import os
import sys
import time
import csv
import string

import numpy

import theano
import theano.tensor as T

import collections, re

feature_size = 0

def find_ngrams(input_list, n):
    return zip(*[input_list[i:] for i in range(n)])

def nlp_prepare(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

#    # Download the MNIST dataset if it is not present
#    data_dir, data_file = os.path.split(dataset)
#    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
#        import urllib
#        origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
#        print 'Downloading data from %s' % origin
#        urllib.urlretrieve(origin, dataset)
#
#    print '... loading data'
#
#
#    # Load the dataset
#    f = gzip.open(dataset, 'rb')
#    train_set, valid_set, test_set = cPickle.load(f)
#    f.close()
#    #train_set, valid_set, test_set format: tuple(input, target)
#    #input is an numpy.ndarray of 2 dimensions (a matrix)
#    #witch row's correspond to an example. target is a
#    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
#    #the number of rows in the input. It should give the target
#    #target to the example with the same index in the input.


    print '... loading data'
    train_set=list();
    valid_set=list();
    test_set=list();
    predict_set=list();

    train_set_size = 6000;
    valid_set_size = 1000;
    test_set_size = 369;
    predict_set_size = 100;

    debug = "false";
    if debug == "true":
      train_set_size = 1000;
      valid_set_size = 200;
      test_set_size = 100;
      predict_set_size = 100;

        
    words_in_doc = {};

    #load data from kaggle test set
    #step1: count word, build word list
    with open('../data/train.tsv', 'rb') as csvfile:
      datareader = csv.reader(csvfile, delimiter='	', quotechar='"')
      for index,row in enumerate(datareader):
        for i in (0,2,3):
          for word in re.findall(r'\w+', row[i]):
            words_in_doc[word]= words_in_doc.get(word,0) + 1;

    with open('../data/test.tsv', 'rb') as csvfile:
      datareader = csv.reader(csvfile, delimiter='	', quotechar='"')
      for index,row in enumerate(datareader):
        for i in (0,2,3):
          for word in re.findall(r'\w+', row[i]):
            words_in_doc[word]= words_in_doc.get(word,0) + 1;

    #make sure word appearency is more than once.
    top_word_list = [word for word in words_in_doc if words_in_doc[word]>10];

    #word indexing
    word2int = {};
    for index,word in enumerate(top_word_list):
        word2int[word] = index;

    word_feature_size = len(top_word_list)*3;
    float_feature_size = 26;
    feature_size = word_feature_size + float_feature_size;

    train_set = [
        numpy.ndarray(shape=(train_set_size, feature_size), dtype=theano.config.floatX),
        numpy.ndarray(shape=(train_set_size), dtype=int)];

    valid_set.append(numpy.ndarray(shape=(valid_set_size, feature_size), dtype=theano.config.floatX));
    valid_set.append(numpy.ndarray(shape=(valid_set_size), dtype=int));
    test_set.append(numpy.ndarray(shape=(test_set_size, feature_size), dtype=theano.config.floatX));
    test_set.append(numpy.ndarray(shape=(test_set_size), dtype=int));
    predict_set.append(numpy.ndarray(shape=(predict_set_size, feature_size), dtype=theano.config.floatX));
    predict_set.append(numpy.ndarray(shape=(predict_set_size), dtype=int));

    def build_feature(row, top_word_list):
      res = numpy.ndarray(shape=(feature_size), dtype=theano.config.floatX);
      
      for i in xrange(feature_size):
        res[i] = 0;

      URL=0;
      CONTENT=2;
      CATEGORY=3;

      #feature from url
      feature_shift = 0;
      words_count = {};
      for word in re.findall(r'\w+', row[URL]):
        words_count[word]= words_count.get(word,0) + 1;

      for word in words_count:
        res[word2int.get(word,0)] = words_count[word];

      #feature from content 
      feature_shift += len(top_word_list);
      words_count = {};
      for word in re.findall(r'\w+', row[CONTENT]):
        words_count[word]= words_count.get(word,0) + 1;

      for word in words_count:
        res[word2int.get(word,0)+feature_shift] = words_count[word];

      #feature from category 
      feature_shift += len(top_word_list);
      words_count = {};
      for word in re.findall(r'\w+', row[CATEGORY]):
        words_count[word]= words_count.get(word,0) + 1;

      for word in words_count:
        res[word2int.get(word,0)+feature_shift] = words_count[word];

      #feature from float points
      feature_shift += len(top_word_list);


      return res;

    #load data from kaggle test set
    with open('../data/train.tsv', 'rb') as csvfile:
      datareader = csv.reader(csvfile, delimiter='	', quotechar='"')
      for index,row in enumerate(datareader):
        if index == train_set_size + valid_set_size + test_set_size : 
          break; 
        if index<train_set_size : 
          #make label
          train_set[1][index] = string.atoi(row[len(row)-1]);
          #feature extractor
          train_set[0][index] = build_feature(row, top_word_list);
        elif index < train_set_size + valid_set_size :
          #make label
          valid_set[1][index-train_set_size] = string.atoi(row[len(row)-1]);
          #feature extractor
          valid_set[0][index-train_set_size] = build_feature(row, top_word_list);
        else :
          #make label
          test_set[1][index-train_set_size-valid_set_size] = string.atoi(row[len(row)-1]);
          #feature extractor
          test_set[0][index-train_set_size-valid_set_size] = build_feature(row, top_word_list);


    print '... loading predict dataset'
    #load data from kaggle test set
    with open('../data/test.tsv', 'rb') as csvfile:
      datareader = csv.reader(csvfile, delimiter='	', quotechar='"')
      for index,row in enumerate(datareader):
        if index == predict_set_size: 
          break;
        if index<train_set_size : 
          #feature extractor
          predict_set[0][index] = build_feature(row, top_word_list);

    train_set = tuple(train_set);
    valid_set = tuple(valid_set);
    test_set = tuple(test_set);
    predict_set = tuple(predict_set);

    data = (train_set, valid_set, test_set, predict_set); 

    print "cPickle dumpping...";
#    file = open(dataset, 'wb')
    file = gzip.GzipFile(dataset, 'wb')
    file.write(cPickle.dumps(data, 1))
    file.close()

    return;

if __name__ == '__main__':
    nlp_prepare("../data/nlp_prepard.pkl.gz")

