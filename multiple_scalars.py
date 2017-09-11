"""
Plotting multiple scalars on the same graph

"""

import tensorflow as tf
from numpy import random

writer_val = tf.summary.FileWriter('./logs/plot_val')
writer_train = tf.summary.FileWriter('./logs/plot_train')

loss_var = tf.Variable(0.0)
tf.summary.scalar("loss", loss_var)

write_op = tf.summary.merge_all()

session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())

for i in range(100):
    # loss validation
    summary = session.run(write_op, {loss_var: random.rand()})
    writer_val.add_summary(summary, i)
    writer_val.flush()

    # loss train
    summary = session.run(write_op, {loss_var: random.rand()})
    writer_train.add_summary(summary, i)
    writer_train.flush()
