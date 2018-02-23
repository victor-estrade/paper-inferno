from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from collections import OrderedDict
import time
from neyman.inferences import batch_hessian
timestr = time.strftime("%Y%m%d-%H%M%S")


tf.flags.DEFINE_string("train_data", default="toy_1D_ind_train_samples.npz",
                       help="path of the training data in npz format")
tf.flags.DEFINE_string("model_dir", default="clf_dir/asimov_{}".format(timestr),
                       help="path to save checkpoints and results")
tf.flags.DEFINE_integer("n_epochs", default=1,
                       help="number of steps for each train call")
tf.flags.DEFINE_integer("batch_size", default=4000, help="")
tf.flags.DEFINE_float("temperature", default=1.0, help="")
tf.flags.DEFINE_boolean("use_cross_entropy", default=False, help="")


flags = tf.flags.FLAGS

ds = tf.contrib.distributions

def small_nn(n_logits=2):
  layers = OrderedDict() 
  activation = tf.nn.relu
  initializer = tf.glorot_normal_initializer
  layers["dense_0"] = tf.layers.Dense(10, activation=activation,
                              kernel_initializer=initializer(),
                              name="dense_0")
  layers["dense_1"] = tf.layers.Dense(10, activation=activation,
                              kernel_initializer=initializer(),
                              name="dense_1")
  layers["output"] = tf.layers.Dense(n_logits, activation=None, name="output")
  return layers


def asimov_model(features, labels, mode, params):

    n_logits = 2
    layers = small_nn(n_logits)

    temperature = tf.convert_to_tensor(params["temperature"], dtype=tf.float32) 
    c_interest  = params["c_interest"] 
    use_cross_entropy  = params["use_cross_entropy"] 

    c_norm_dists = {"bkg" : ds.Normal(loc=400., scale=20.)}

    if "components" in features:
      # get name of each component
      c_names = sorted(features["components"].keys())
      c_tensors = [features["components"][c_name] for c_name in c_names] 
      c_sizes = [tf.shape(c_tensor)[0] for c_tensor in c_tensors]
      if "c_norms" in features:
        c_norms = [features["c_norms"][c_name]  for c_name in c_names]
      for c_name, c_size in zip(c_names, c_sizes):
        tf.summary.scalar("c_batch_size_{}".format(c_name), c_size)
      X = tf.concat(c_tensors, axis=0, name="concat_components")
      y = tf.concat([tf.ones((c_size,),dtype=tf.int32)*i 
        for i, c_size in enumerate(c_sizes)], axis=0) 
    else:
      X = features["X"]
      y = labels 

    inputs = tf.reshape(X, (-1,1))

    net = inputs
    for name, layer in layers.items():
      net = layer(net) 
      for var in layer.variables:
        tf.summary.histogram("h_{}".format(var.name), var)
    logits = net

    probs = tf.nn.softmax(logits/temperature)

    if mode == tf.estimator.ModeKeys.PREDICT:
      predictions = {"probabilities" : probs}
      return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    split_probs = tf.split(probs, c_sizes, name="split_probs")
    split_means = [tf.reduce_mean(split_prob, axis=0) for split_prob in split_probs]

    mu = tf.convert_to_tensor(1., name="mu")
    split_counts = [ mean*norm*mu if (c_name==c_interest) else mean*norm
        for c_name, mean, norm in zip(c_names, split_means, c_norms)]

    for mean, count, c_name in zip(split_means, split_counts, c_names):
      for n in range(n_logits):
        tf.summary.scalar("mean_{}_{}".format(c_name,n), mean[n])
        tf.summary.scalar("count_{}_{}".format(c_name,n), count[n])
       
    exp_counts = sum(split_counts) 

    # asimov loss
    nuis_pars = [] 
    with tf.name_scope("compute_asimov_loss"):
      pois = ds.Poisson(exp_counts, name="poisson")
      asimov = tf.stop_gradient(exp_counts, name="asimov")
      ll = tf.reduce_sum(pois.log_prob(asimov), name="likelihood")
      # add c_norm constrain terms
      constraint_terms = [] 
      for c_norm, c_name in zip(c_norms, c_names):
        if c_name in c_norm_dists: 
          nuis_pars.append(c_norm)
          constraint = c_norm_dists[c_name]
          constraint_terms.append(tf.reduce_sum(constraint.log_prob(c_norm),
            name="c_norm_{}_log_prob".format(c_name)))
      nll = - tf.reduce_sum([ll]+constraint_terms)
      hess = batch_hessian(nll, [mu]+nuis_pars)
      cov = tf.matrix_inverse(hess[0])
      asimov_loss = tf.sqrt(cov[0,0]) 

    cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y,
                                                           logits=logits)

    tf.summary.scalar("asimov_loss", asimov_loss) 
    tf.summary.scalar("cross_entropy", cross_entropy)

    loss = cross_entropy if use_cross_entropy else asimov_loss 

    if mode == tf.estimator.ModeKeys.EVAL:

      return tf.estimator.EstimatorSpec( mode, loss=loss )

    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)    

def main(_):

  data = np.load(flags.train_data)


  params = { "temperature" : flags.temperature,
             "c_interest" : "sig",
             "use_cross_entropy" : flags.use_cross_entropy}

  config = tf.estimator.RunConfig(save_summary_steps=20)

  clf = tf.estimator.Estimator(model_fn=asimov_model, model_dir=flags.model_dir,
                               params=params, config=config)

  print(clf.model_dir)

  train_data = {}
  val_data = {}
  for key, value in data.items():
    train_data[key], val_data[key] = train_test_split(value) 

  print(data.keys())

  def make_input_fn(data):
    def input_fn():

      components = {}
      for key, value in data.items():
        components[key] = tf.data.Dataset.from_tensor_slices(value)\
                               .shuffle(buffer_size=10000)\
                               .batch(flags.batch_size)

      c_norms = tf.data.Dataset.from_tensors({"sig" : 20., "bkg" : 400.}).repeat()

      dataset = tf.data.Dataset.zip({"components" : components, "c_norms" : c_norms})

      dataset_it = dataset.make_one_shot_iterator()
      next_batch = dataset_it.get_next()

      return next_batch, None 
    return input_fn
    

  train_input_fn = make_input_fn(train_data)
  val_input_fn = make_input_fn(val_data)

  for i in range(flags.n_epochs):
    clf.train(input_fn=train_input_fn)
    clf.evaluate(input_fn=val_input_fn)

if __name__ == "__main__":
  tf.app.run()
