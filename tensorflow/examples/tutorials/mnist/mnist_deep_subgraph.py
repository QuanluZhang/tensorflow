# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A deep MNIST classifier using convolutional layers.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

import numpy as np
import time
import timeit

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def deepnn(x):
  """deepnn builds the graph for a deep net for classifying digits.

  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.

  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 28, 28, 1])

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 2X.
  with tf.name_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # Second pooling layer.
  with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2)

  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features.
  with tf.name_scope('fc1'):
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 1024 features to 10 classes, one for each digit
  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  return y_conv, keep_prob


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def main(argv):

  subgraph_file_path = argv[1]

  tf.reset_default_graph()
  tf.train.import_meta_graph('/tmp/mymodel.meta')
  accuracy = tf.get_collection('my_accuracy')[0]
  train_step = tf.get_collection('my_train_step')[0]
  [x, y_, keep_prob] = tf.get_collection('inputs')

  # change graph def
  name_type_map = dict()
  #subg_fd = open("/home/quzha/work/GraphPartition/subgraph_nodes.csv", "r")
  subg_fd = open(subgraph_file_path, "r")
  line = subg_fd.readline()
  segs = line.split()
  assert(segs[0] == "input_nodes")
  subg_inputs = []
  for i in range(int(segs[1])):
    line = subg_fd.readline()
    full_name = line.split(";")[1]
    subg_inputs.append(full_name)
    name_type_map[full_name] = line.split(";")[2][:-1]

  line = subg_fd.readline()
  segs = line.split()
  assert(segs[0] == "true_nodes")
  subg_nodes = []
  for i in range(int(segs[1])):
    line = subg_fd.readline()
    full_name = line.split(";")[1]
    subg_nodes.append(full_name)
    name_type_map[full_name] = line.split(";")[2][:-1]

  line = subg_fd.readline()
  segs = line.split()
  assert(segs[0] == "boundary_nodes")
  subg_bd_nodes = []
  for i in range(int(segs[1])):
    line = subg_fd.readline()
    full_name = line.split(";")[1]
    subg_bd_nodes.append(full_name)
    name_type_map[full_name] = line.split(";")[2][:-1]

  line = subg_fd.readline()
  segs = line.split()
  assert(segs[0] == "sink_node")
  line = subg_fd.readline()
  subg_sink = line.split(";")[1]
  name_type_map[full_name] = line.split(";")[2][:-1]
  subg_fd.close()

  # get sink op
  sink_op = tf.get_default_graph().get_operation_by_name(subg_sink)
  #print(dir(sink_op))
  #return

  # load customized op
  tg_module = tf.load_op_library('/home/quzha/work/tensorflow/tensorflow/core/user_ops/tensor_generator.so')
  #print(dir(tg_module))

  sys.path.append('/home/quzha/work/GraphPartition/')
  import parse_runtime_shape
  runtime_shape = parse_runtime_shape.parse_runtime_shape('/home/quzha/static_analysis/result/dump_output_shape.txt')
  #print(runtime_shape)

  # create placeholder/constant for input node
  new_operations = dict()
  for each in subg_inputs:
    if name_type_map[each] == "_Arg":
      continue
    if name_type_map[each] == "NoOp":
      noop = tf.no_op()
      assert(not each in new_operations)
      new_operations[each] = [noop]
      print("-------------------------NoOp-----------------------------")
      continue
    op = tf.get_default_graph().get_operation_by_name(each)
    for i in range(len(op.outputs)):
      #print(len(op.outputs), op.outputs[0])
      print("subg_inputs: ", each, " shape: ", op.outputs[i].shape)
      if name_type_map[each] == "Const":
        assert(op.outputs[i].shape.is_fully_defined())
        new_const = tf.identity(op.outputs[i])
        if each in new_operations:
          new_operations[each].append(new_const)
        else:
          new_operations[each] = [new_const]
      else:
        if not op.outputs[i].shape.is_fully_defined():
          assert(each in runtime_shape)
          assert(len(runtime_shape[each]) > i)
          assert(name_type_map[each] != "VariableV2")
          input_array = [int(op.outputs[i].dtype)]
          input_array.extend(runtime_shape[each][i])
          new_op = tg_module.tensor_generator_tma(input_array, op.outputs[i].dtype, tensor_name = each)
        #elif each == "fc1/Variable_1":
        elif name_type_map[each] == "VariableV2":
          assert(op.outputs[0].shape.is_fully_defined())
          new_op = tf.Variable(tf.zeros(op.outputs[0].shape), dtype = tf.float32, expected_shape = op.outputs[0].shape)
        else:
          assert(op.outputs[0].shape.is_fully_defined())
          tmp_array = []
          tmp_array.append(int(op.outputs[0].dtype))
          for dim in op.outputs[0].shape:
            tmp_array.append(int(dim))
          new_op = tg_module.tensor_generator_tma(tmp_array, op.outputs[0].dtype, tensor_name = each)
          #new_operations[each] = new_op
        if each in new_operations:
          new_operations[each].append(new_op)
        else:
          new_operations[each] = [new_op]

  # update boundary nodes' inputs
  new_pholders = dict()
  for each in subg_bd_nodes:
    op = tf.get_default_graph().get_operation_by_name(each)
    for i in range(len(op.inputs)):
      #print("**************************************************************")
      #print(op.inputs[i].dtype)
      #print(dir(op.inputs[i].dtype))
      #print(op.inputs[i].dtype.name)
      #print(dir(op.inputs[i]))
#      if len(op.inputs[i].dtype.name) >= 3 and op.inputs[i].dtype.name[-3:] == 'ref':
#        print("**************************************************************")
#        if op.inputs[i].name[:-2] in new_operations:
#          print("abc")
#          if name_type_map[op.inputs[i].name[:-2]] == "VariableV2":
#            print("def")
#            print(type(new_operations[op.inputs[i].name[:-2]][output_index]))
#            print(dir(new_operations[op.inputs[i].name[:-2]][output_index]))
#            print(type(new_operations[op.inputs[i].name[:-2]][output_index]._ref()))
#            print(dir(new_operations[op.inputs[i].name[:-2]][output_index]._ref()))
#            print(type(new_operations[op.inputs[i].name[:-2]][output_index].read_value()))
#            print(dir(new_operations[op.inputs[i].name[:-2]][output_index].read_value()))
#          else:
#            print("def2")
#        else:
#          print("abc2")
      # TODO: it is possible that :XX rather than :X
      assert(len(op.inputs[i].name.split(':')[-1]) == 1)
      output_index = int(op.inputs[i].name.split(':')[-1])
      if op.inputs[i].name[:-2] in new_operations:
        #if op.inputs[i].name[:-2] == "fc1/Variable_1":
        if name_type_map[op.inputs[i].name[:-2]] == "VariableV2":
          #print("VariableV2: ", op.inputs[i].name)
          assert(output_index == 0)
          if len(op.inputs[i].dtype.name) >= 3 and op.inputs[i].dtype.name[-3:] == 'ref':
            op._update_input(i, new_operations[op.inputs[i].name[:-2]][output_index]._ref())
          else:
            op._update_input(i, new_operations[op.inputs[i].name[:-2]][output_index].read_value())
        else:
          op._update_input(i, new_operations[op.inputs[i].name[:-2]][output_index])
      else:
        # if the input is not in new_operations, and its predecessor is a placeholder (Note: or NoOp),
        # we still need to record it in order to feed data
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX", op.inputs[i].name[:-2])
        predecessor_op = tf.get_default_graph().get_operation_by_name(op.inputs[i].name[:-2])
        if predecessor_op.type == 'Placeholder':
          new_pholders[op.inputs[i].name[:-2]] = op.inputs[i]

  # update control inputs
  for each in subg_bd_nodes:
    op = tf.get_default_graph().get_operation_by_name(each)
    rm_num = op._remove_control_input(new_operations)
    if rm_num > 0:
      print("******************NoOp********************: ", rm_num)

  # get placeholder in subg_nodes(true_nodes), and put them in new_pholders
  existing_pholder_in_subg = dict()
  for each in subg_nodes:
    if name_type_map[each] == "_Arg":
      #existing_pholder_in_subg[each] = tf.get_default_graph().get_operation_by_name(each[5:-4])
      segs = each.split('_')
      pholder_tensor_name = '_'.join(segs[2:-2])
      existing_pholder_in_subg[each] = tf.get_default_graph().get_tensor_by_name(pholder_tensor_name+":0")

  with tf.Session() as sess:
    feed_data = dict()
    for name in new_pholders:
      if (name in name_type_map) and (name_type_map[name] == "Const"):
        assert(False)
        continue
      assert(new_pholders[name].shape.dims != None)
      #if new_pholders[name].shape.dims == None:
      #  print("unknown shape: ", new_pholders[name].shape.dims)
      #  feed_data[new_pholders[name]] = np.ndarray(shape = (1), dtype = new_pholders[name].dtype.as_numpy_dtype)
      #else:
      #  feed_data[new_pholders[name]] = np.ndarray(shape = new_pholders[name].shape, dtype = new_pholders[name].dtype.as_numpy_dtype)
      feed_data[new_pholders[name]] = np.ndarray(shape = new_pholders[name].shape, dtype = new_pholders[name].dtype.as_numpy_dtype)
    for name in existing_pholder_in_subg:
      print("runtime_shape: ", runtime_shape[name][0])
      feed_data[existing_pholder_in_subg[name]] = np.ndarray(shape = runtime_shape[name][0], dtype = float)
    sess.run(tf.global_variables_initializer())
    for i in range(10):
      sess.run(sink_op, feed_dict = feed_data)
    print("warm up done")
    start = time.clock()
    for i in range(1000):
      sess.run(sink_op, feed_dict = feed_data)
    res_duration = (time.clock() - start) * 1000 / 1000
    print("duration: ", res_duration, "ms")
    return res_duration

    #timeit.timeit(sess.run(sink_op, feed_dict = feed_data), number=1000)

def evaluate_subgraph(subgraph_file_path):
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  #res = tf.app.run(main=main, argv=[sys.argv[0]] + [subgraph_file_path] + unparsed)
  res = main(argv=[sys.argv[0]] + [subgraph_file_path])
  return res

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  #tf.app.run(main=main, argv=[sys.argv[0]] + ["/home/quzha/work/GraphPartition/subgraph_nodes.csv"] + unparsed)
  #tf.app.run(main=main, argv=[sys.argv[0]] + ["/home/quzha/work/GraphPartition/one_branch_subgraph.csv"] + unparsed)
  #tf.app.run(main=main, argv=[sys.argv[0]] + ["/home/quzha/work/GraphPartition/failed_cases/one_branch_subgraph.csv1517651679.4963002"] + unparsed)
  tf.app.run(main=main, argv=[sys.argv[0]] + ["/home/quzha/work/GraphPartition/failed_cases/one_branch_subgraph.csv1517651686.2848566"] + unparsed)
