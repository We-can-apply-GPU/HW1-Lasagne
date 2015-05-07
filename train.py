#!/usr/bin/env python

from __future__ import print_function
import sys
import time

import numpy as np
import theano
import theano.tensor as T
import lasagne
import itertools

#User defined
from userCode import iofile
from userCode import util
from settings import *

def load_data():
  fbank = iofile.read_fbank(sys.argv[1])
  label = iofile.read_label(sys.argv[1])
  X_train = np.array([data[1:] for data in fbank]).astype(theano.config.floatX)
  Y_train = np.array([data[1] for data in label]).astype('int32')
  return dict(
    X_train = theano.shared(X_train),
    Y_train = theano.shared(Y_train),
    num_train = X_train.shape[0],
    input_dim = X_train.shape[1],
    output_dim = 48
  )


#Todo
#->Add weight initialize,need readWeight
def build_model(input_dim, output_dim, batch_size=BATCH_SIZE,parsPath=""):
    #if(parsPath == ""):
        #weightsLs
        #biasesLs
    ##Input Layer

    NNlayers=[]
    print("Build Layers of network...")
    
    ##Input Layer
    l_in = lasagne.layers.InputLayer(shape=(BATCH_SIZE, input_dim),name="InputLayer")
    NNlayers.append(l_in)

    ##Hidden Layer and Dropout Layer
    for curLayerIndex in range(1,len(LAYERS)-1):
        NNlayer = lasagne.layers.DenseLayer(
                NNlayer[(curLayerIndex-1) * 2],
                num_units=Layer[curLayerIndex],
                W = lasagne.init.Normal(0.01),
                nonlinearit=lasagne.nonlinearities.rectify)
        NNlayers.append(NNlayer)

        dpLayer = lasagne.layers.DropoutLayer(NNlayers[(curLayerIndex-1)*2+1],rescale=True, p=0.1)
        NNlayer.append(dpLayer)

    ##Output Layer
    l_out = lasagne.layers.DenseLayer(
            NNlayers[-1],
            num_units=output_dim,
            nonlinearity=lasagne.nonlinearities.softmax
            )
    return l_out

def create_iter_functions(data, output_layer):

    batch_index = T.iscalar('batch_index')
    x_batch = T.matrix('x')
    y_batch = T.ivector('y')

    batch_slice = slice(batch_index * BATCH_SIZE, (batch_index+1) * BATCH_SIZE)
    objective = lasagne.objectives.Objective(output_layer, loss_function=lasagne.objectives.categorical_crossentropy)

    loss_train = objective.get_loss(x_batch, target=y_batch)
    loss_eval = objective.get_loss(x_batch, target=y_batch, deterministic=True)

    pred = T.argmax(output_layer.get_output(x_batch, deterministic=True), axis=1)
    accuracy = T.mean(T.eq(pred, y_batch), dtype=theano.config.floatX)

    all_params = lasagne.layers.get_all_params(output_layer)
    updates = lasagne.updates.nesterov_momentum(loss_train, all_params, LEARNING_RATE, MOMENTUM)

    iter_train = theano.function(
          [batch_index], loss_train, updates=updates,
          givens={
              x_batch: data['X_train'][batch_slice],
              y_batch: data['Y_train'][batch_slice]
              }
          )

    iter_valid = theano.function(
          [batch_index], [loss_eval, accuracy],
          givens={
              x_batch: data['X_train'][batch_slice],
              y_batch: data['Y_train'][batch_slice]
              }
          )

    return dict(
          train=iter_train,
          valid=iter_valid
          )

def main():
    print("Loading data...")
    data = load_data()
    print("Building model and compile theano...")
    print(data['num_train'])
    output_layer = build_model(input_dim = data['input_dim'], output_dim = data['output_dim'])
    iter_funcs = create_iter_functions(data, output_layer)
    print("Training")
    now = time.time()
    try:
        for epoch in range(NUM_EPOCHS):
            num_batches_train = data['num_train'] // BATCH_SIZE
            batch_train_losses = []
        for b in range(num_batches_train):
            batch_train_loss = iter_funcs['train'](b)
            batch_train_losses.append(batch_train_loss)
        avg_train_loss = np.mean(batch_train_losses)
        print("Epoch {} of {} took {:.3f}s".format(epoch+1, NUM_EPOCHS, time.time() - now))
        print("  training loss:\t\t{:.6f}".format(avg_train_loss))
        if epoch % 10 == 0:
            batch_valid_accus = []
            batch_valid_losses = []
        for b in range(num_batches_train):
            batch_valid_loss, batch_valid_accu = iter_funcs['valid'](b)
            batch_valid_losses.append(batch_valid_loss)
            batch_valid_accus.append(batch_valid_accu)
        avg_valid_loss = np.mean(batch_valid_losses)
        avg_valid_accu = np.mean(batch_valid_accus)
        print("--validation loss:\t\t{:.2f}".format(avg_valid_loss))
        print("--validation accuracy:\t\t{:.2f} %".format(avg_valid_accu * 100))
        import pickle
        fout = open("model/5d/{:.2f}".format(avg_valid_accu * 100), "w")
        pickle.dump(lasagne.layers.get_all_param_values(output_layer), fout)
        now = time.time()

    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
   main() 
