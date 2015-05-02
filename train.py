#!/usr/bin/env python

from __future__ import print_function
import sys
import numpy as np
import theano
import theano.tensor as T
import lasagne
import util
import time

BATCH_SIZE = 100
NUM_HIDDEN_UNITS = 128
LEARNING_RATE = 0.01
MOMENTUM = 0.9
NUM_EPOCHS = 500

def load_data():
  fbank = util.read_fbank(sys.argv[1])
  label = util.read_label(sys.argv[1])
  print(label)
  X_train = np.array([data[1:] for data in fbank]).astype(theano.config.floatX)
  Y_train = np.array([data[1] for data in label]).astype('int32')
  return dict(
    X_train = theano.shared(X_train),
    Y_train = theano.shared(Y_train),
    num_examples_train = X_train.shape[0],
    input_dim = X_train.shape[1],
    output_dim = 48
  )

def build_model(input_dim, output_dim, batch_size=BATCH_SIZE, num_hidden_units=NUM_HIDDEN_UNITS):
  l_in = lasagne.layers.InputLayer(
    shape=(batch_size, input_dim)
  )
  l_hidden1 = lasagne.layers.DenseLayer(
    l_in,
    num_units=num_hidden_units,
    nonlinearity=lasagne.nonlinearities.rectify
  )
  l_hidden1_dropout = lasagne.layers.DropoutLayer(
    l_hidden1,
    p=0.5
  )
  l_hidden2 = lasagne.layers.DenseLayer(
    l_in,
    num_units=num_hidden_units,
    nonlinearity=lasagne.nonlinearities.rectify
  )
  l_hidden2_dropout = lasagne.layers.DropoutLayer(
    l_hidden2,
    p=0.5
  )
  l_out = lasagne.layers.DenseLayer(
    l_hidden2_dropout,
    num_units=output_dim,
    nonlinearity=lasagne.nonlinearities.softmax
  )
  return l_out

def create_iter_functions(data, output_layer, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, momentum=MOMENTUM):
  print(learning_rate, momentum)
  batch_index = T.iscalar('batch_index')
  x_batch = T.matrix('x')
  y_batch = T.ivector('y')

  batch_slice = slice(batch_index * batch_size, (batch_index+1) * batch_size)
  objective = lasagne.objectives.Objective(output_layer, loss_function=lasagne.objectives.categorical_crossentropy)

  loss_train = objective.get_loss(x_batch, target=y_batch)
  all_params = lasagne.layers.get_all_params(output_layer)
  updates = lasagne.updates.nesterov_momentum(loss_train, all_params, learning_rate, momentum)

  iter_train = theano.function(
    [batch_index], loss_train, updates=updates,
    givens={
      x_batch: data['X_train'][batch_slice],
      y_batch: data["Y_train"][batch_slice]
    }
  )

  return dict(
    train=iter_train
  )

def train(iter_funcs, dataset, batch_size=BATCH_SIZE):
  num_batches_train = dataset['num_examples_train'] // batch_size
  epoch = 1
  while True:
    batch_train_losses = []
    for b in range(num_batches_train):
      batch_train_loss = iter_funcs['train'](b)
      batch_train_losses.append(batch_train_losses)
    avg_train_loss = np.mean(batch_train_losses)
    yield {
      'number': epoch,
      'train_loss': avg_train_loss
    }
    epoch += 1


def main(num_epochs=NUM_EPOCHS):
  print("Loading data...")
  data = load_data()
  print(data)
  print("Building model and compile theano...")
  output_layer = build_model(input_dim = data['input_dim'], output_dim = data['output_dim'])
  iter_funcs = create_iter_functions(data, output_layer)
  print("Training")
  now = time.time()
  #try:
  for epoch in train(iter_funcs, data):
    print("Epoch {} of {} took {:.3f}s".format(epoch['number'], num_epochs, time.time() - now))
    now = time.time
    print("  training loss:\t\t{:.6f}".format(epoch['train_loss']))
    if epoch['number'] >= num_epochs:
      break;
  """
  except KeyboardInterrupt:
    pass
  """


if __name__ == '__main__':
  main()
