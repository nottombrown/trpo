import os.path as osp

import numpy as np
import tensorflow as tf

CLIP_LENGTH = 1.5

def add_simple_summary(summary_writer, tag, simple_value, step):
    summary_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=simple_value)]), step)

class TBLogger():
    def __init__(self, name):
        logs_path = osp.expanduser('~/tb/comparison/ilyasu/%s' % (name))
        self.summary_writer = tf.summary.FileWriter(logs_path)
        self.summary_step = 0

    def log(self, tag, simple_value):
        add_simple_summary(self.summary_writer, tag, simple_value, self.summary_step)
