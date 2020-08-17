seed_value= 0
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
import tensorflow as tf
tf.compat.v1.set_random_seed(seed_value)


def my_init(shape,dtype=float):
	return tf.random.normal(shape,dtype=dtype)

print(my_init((2,2)))