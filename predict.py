#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import util
import lasagne
import pickle
import theano
import theano.tensor as T
import sys

NUM_HIDDEN_UNITS = 512
DATA = "train"

def build_model(input_dim, output_dim, num_hidden_units=NUM_HIDDEN_UNITS):
  l_in = lasagne.layers.InputLayer(
    shape=(None, input_dim)
  )
  l_hidden1 = lasagne.layers.DenseLayer(
    l_in,
    num_units=num_hidden_units,
    nonlinearity=lasagne.nonlinearities.rectify
  )
  l_dp1 = lasagne.layers.DropoutLayer(l_hidden1, rescale=True, p=0.1)
  l_hidden2 = lasagne.layers.DenseLayer(
    l_dp1,
    num_units=num_hidden_units,
    nonlinearity=lasagne.nonlinearities.rectify
  )
  l_dp2 = lasagne.layers.DropoutLayer(l_hidden2, rescale=True, p=0.1)
  l_hidden3 = lasagne.layers.DenseLayer(
    l_dp2,
    num_units=num_hidden_units,
    nonlinearity=lasagne.nonlinearities.rectify
  )
  l_dp3 = lasagne.layers.DropoutLayer(l_hidden3, rescale=True, p=0.1)
  l_hidden4 = lasagne.layers.DenseLayer(
    l_dp3,
    num_units=num_hidden_units,
    nonlinearity=lasagne.nonlinearities.rectify
  )
  l_dp4 = lasagne.layers.DropoutLayer(l_hidden4, rescale=True, p=0.1)
  l_hidden5 = lasagne.layers.DenseLayer(
    l_dp4,
    num_units=num_hidden_units,
    nonlinearity=lasagne.nonlinearities.rectify
  )
  l_dp5 = lasagne.layers.DropoutLayer(l_hidden5, rescale=True, p=0.1)
  l_out = lasagne.layers.DenseLayer(
    l_dp5,
    num_units=output_dim,
    nonlinearity=lasagne.nonlinearities.softmax
  )
  return l_out

def printtrim(ans, fout):
    for start in range(len(ans)):
        if ans[start] != 37:
            break
    for end in range(len(ans)-1, -1, -1):
        if ans[end] != 37:
            break
    pre = -1
    for index in range(start, end+1):
        if ans[index] == pre:
            continue
        print(util.index2ans[ans[index]], file=fout, end='')
        pre = ans[index]
    print(file=fout)

def main():
    print("Loading data...")
    fbank = util.read_fbank(DATA)
    data = np.array([data[1:] for data in fbank]).astype(theano.config.floatX)
    output_layer = build_model(input_dim=69, output_dim=48)
    lasagne.layers.set_all_param_values(output_layer, pickle.load(open("model/5d/"+sys.argv[1], "r")))
    x = T.matrix('x')
    predict = theano.function([x], T.argmax(output_layer.get_output(x, deterministic=True), axis=1))
    fout = open(DATA + sys.argv[1]+".ans", "w")
    print("id,phone_sequence", file=fout)
    print("{}_{},".format(fbank[0][0][0], fbank[0][0][1]), file=fout, end='')
    pre = 0
    for f, index in zip(fbank, range(len(fbank))):
        if f[0][2] == 1 and index != 0:
            ans = predict(data[pre:index])
            printtrim(ans, fout)
            pre = index
            print("{}_{},".format(f[0][0], f[0][1]), file=fout, end='')
    ans = predict(data[pre:])
    printtrim(ans, fout)
    fout.close()

if __name__ == '__main__':
    main()

