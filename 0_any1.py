import tensorflow as tf
import numpy as np

print("TF:", tf.__version__)
print("NP:", np.__version__)
print("OK:", tf.reduce_sum(tf.constant([[1.,2.],[3.,4.]])).numpy())
