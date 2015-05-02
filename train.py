#!/usr/bin/env python

from __future__ import print_function
import sys
import numpy as np
import theano
import theano.tensor as T
import lasagne
import util
import time
import itertools

BATCH_SIZE = 1000
NUM_HIDDEN_UNITS = 96
LEARNING_RATE = 0.01
MOMENTUM = 0.9
NUM_EPOCHS = 5000000

def load_data():
  fbank = util.read_fbank(sys.argv[1])
  label = util.read_label(sys.argv[1])
  X_train = np.array([data[1:] for data in fbank]).astype(theano.config.floatX)
  Y_train = np.array([data[1] for data in label]).astype('int32')
  return dict(
    X_train = theano.shared(X_train),
    Y_train = theano.shared(Y_train),
    num_train = X_train.shape[0],
    input_dim = X_train.shape[1],
    output_dim = 48
  )

def build_model(input_dim, output_dim, batch_size=BATCH_SIZE, num_hidden_units=NUM_HIDDEN_UNITS):
  l_in = lasagne.layers.InputLayer(
    shape=(None, input_dim)
  )
  l_hidden1 = lasagne.layers.DenseLayer(
    l_in,
    num_units=num_hidden_units,
    nonlinearity=lasagne.nonlinearities.rectify
  )
  l_hidden2 = lasagne.layers.DenseLayer(
    l_hidden1,
    num_units=num_hidden_units,
    nonlinearity=lasagne.nonlinearities.rectify
  )
  l_hidden3 = lasagne.layers.DenseLayer(
    l_hidden2,
    num_units=num_hidden_units,
    nonlinearity=lasagne.nonlinearities.rectify
  )
  l_hidden4 = lasagne.layers.DenseLayer(
    l_hidden3,
    num_units=num_hidden_units,
    nonlinearity=lasagne.nonlinearities.rectify
  )
  l_hidden5 = lasagne.layers.DenseLayer(
    l_hidden4,
    num_units=num_hidden_units,
    nonlinearity=lasagne.nonlinearities.rectify
  )
  l_out = lasagne.layers.DenseLayer(
    l_hidden5,
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
    [], [loss_eval, accuracy],
    givens={
      x_batch: data['X_train'],
      y_batch: data['Y_train']
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
        valid_loss, valid_accuracy = iter_funcs['valid']()
        print("  validation loss:\t\t{:.6f}".format(float(valid_loss)))
        print("  validation accuracy:\t\t{:.2f} %".format(valid_accuracy * 100))
      now = time.time()

  except KeyboardInterrupt:
    pass

if __name__ == '__main__':
  main()
