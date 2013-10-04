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
    predict_set_size = 2800;

    debug = "false";
    if debug == "true":
      train_set_size = 3600;
      valid_set_size = 500;
      test_set_size = 100;
      predict_set_size = 2800;

        
    words_in_doc = {};

    #load data from kaggle test set
    #step1: count word
    with open('../data/train.tsv', 'rb') as csvfile:
      datareader = csv.reader(csvfile, delimiter='	', quotechar='"')
      for index,row in enumerate(datareader):
        bagsofwords = [];
        for i in (0,2,3):
          for word in re.findall(r'\w+', row[i]):
            words_in_doc[word]= words_in_doc.get(word,0) + 1;

    with open('../data/test.tsv', 'rb') as csvfile:
      datareader = csv.reader(csvfile, delimiter='	', quotechar='"')
      for index,row in enumerate(datareader):
        bagsofwords = [];
        for i in (0,2,3):
          for word in re.findall(r'\w+', row[i]):
            words_in_doc[word]= words_in_doc.get(word,0) + 1;

#    top_word = sorted(words_in_doc.items(), key=lambda x: x[1], reverse=1);
    top_word_list = [word for word in words_in_doc if words_in_doc[word]>1];
    print top_word_list;
    print len(top_word_list);

    train_set = [
        numpy.ndarray(shape=(train_set_size,28*28), dtype=theano.config.floatX),
        numpy.ndarray(shape=(train_set_size), dtype=int)];

    valid_set.append(numpy.ndarray(shape=(valid_set_size,28*28), dtype=theano.config.floatX));
    valid_set.append(numpy.ndarray(shape=(valid_set_size), dtype=int));
    test_set.append(numpy.ndarray(shape=(test_set_size,28*28), dtype=theano.config.floatX));
    test_set.append(numpy.ndarray(shape=(test_set_size), dtype=int));
    predict_set.append(numpy.ndarray(shape=(predict_set_size,28*28), dtype=theano.config.floatX));
    predict_set.append(numpy.ndarray(shape=(predict_set_size), dtype=int));

    #load data from kaggle test set
    with open('../data/train.tsv', 'rb') as csvfile:
      datareader = csv.reader(csvfile, delimiter='	', quotechar='"')
      for index,row in enumerate(datareader):
        if index<train_set_size : 
          train_set[1][index] = string.atoi(row[len(row)-1]);

#      index=0;
#      for row in datareader:
#        if index<train_set_size : 
#          train_set[1][index] = string.atoi(row[0]);
#          for pixel_index in xrange(1,28*28+1) : 
#            train_set[0][index][pixel_index-1] = string.atof(row[pixel_index])/255;
#        elif index < train_set_size + valid_set_size :
#          valid_set[1][index-train_set_size] = string.atoi(row[0]);
#          for pixel_index in xrange(1,28*28+1) : 
#            valid_set[0][index-train_set_size][pixel_index-1] = string.atof(row[pixel_index])/255;
#        else :
#          test_set[1][index-train_set_size-valid_set_size] = string.atoi(row[0]);
#          for pixel_index in xrange(1,28*28+1) : 
#            test_set[0][index-train_set_size-valid_set_size][pixel_index-1] = string.atof(row[pixel_index])/255;
#        index+=1;
#        if index == train_set_size + valid_set_size + test_set_size : 
#          break; 
    
    print '... loading predict dataset'
    #load data from kaggle test set
    with open('../data/test.csv', 'rb') as csvfile:
      datareader = csv.reader(csvfile, delimiter=',')
#      index=0;
#      for row in datareader:
#        for pixel_index in xrange(0,28*28) : 
#          predict_set[0][index][pixel_index] = string.atof(row[pixel_index])/255;
#        index+=1;
#        if index == predict_set_size: 
#          break;

    train_set = tuple(train_set);
    valid_set = tuple(valid_set);
    test_set = tuple(test_set);
    predict_set = tuple(predict_set);

    data = (train_set, valid_set, test_set, predict_set); 

    file = gzip.GzipFile(dataset, 'wb')
    file.write(cPickle.dumps(data, 1))
    file.close()

    return;

if __name__ == '__main__':
    nlp_prepare("../data/nlp_prepard.pkl.gz")

