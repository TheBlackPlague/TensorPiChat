# TensorPiChat
# A basic chat-bot using sequence-based learning using TensorFlow. Datastructure used is "Cornell's Movie Dialog Corpus".
# Licensed under MIT - Developed by Shaheryar Sohail. By the MIT License, you can do whatever with any code from this file as long as you agree to have provide a commented in citation of original source.

import sys
import os
import random
import time
import numpy
import tensorflow
import configuration
import datamanagement
from datamodel import TensorPiDM
import argparse

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def gh():
    th = datamanagement.load_d("tid.enc", "tid.dec")
    dh = datamanagement.load_d("train.enc", "train.dec")
    training_data_holder_size = [
        len(dh[d]) for d in range(len(configuration.DATA_HOLDER))
    ]
    training_data_holder_maxsize = sum(training_data_holder_size)
    training_data_holder_scale = [
        sum(training_data_holder_size[:i + 1]) / training_data_holder_maxsize
        for i in range(len(training_data_holder_size))
    ]
    print("Samples per holder:\n", training_data_holder_size)
    print("Scale per holder:\n", training_data_holder_scale)
    return th, dh, training_data_holder_scale

def grh(training_data_holder_scale):
    randomcheck = random.random()
    return min([
        i for i in range(len(training_data_holder_scale))
        if training_data_holder_scale[i] > randomcheck
    ])

def value_lenght_checker(es, ds, ei, di, dm):
    if len(ei) != es:
        raise ValueError("Encoder lenght error... equality disturbed,"
        " %d != %d." % (len(ei), es))
    if len(di) != ds:
        raise ValueError("Decoder lenght error... equality disturbed,"
        " %d != %d." % (len(di), ds))
    if len(dm) != ds:
        raise ValueError("Weighting lenght error... equality disturbed,"
        " %d != %d." % (len(dm), ds))

def do_training_step(s, model, ei, di, dm, hid, fwo):
    es, ds = configuration.DATA_HOLDER[hid]
    value_lenght_checker(ei, ds, ei, di, dm)
    i_feed = {}
    for t_step in range(es):
        i_feed[model.ei[t_step].name] = ei[t_step]
    for t_step in range(ds):
        # Set for both decoder and weights
        i_feed[model.di[t_step].name] = di[t_step]
        i_feed[model.dm[t_step].name] = dm[t_step]
    lt = model.di[ds].name
    i_feed[lt] = numpy.zeros([
        model.bs
    ], dtype = numpy.int32)
    if not fwo:
        o_feed = [
            model.train_operative[hid],
            model.gradient_normal[hid],
            model.loss[hid]
        ]
    else:
        o_feed = [
            model.loss[hid]
        ]
        for t_step in range(ds):
            o_feed.append(model.output[hid][t_step])
    output = s.run(o_feed, i_feed)
    if not fwo:
        return output[1], output[2], None
    else:
        return None, output[0], output[1:]

def get_skipped_training_step(iteration):
    if iteration < 100:
        return 30
    return 100

def restore(s, save_instance):
    checkpoint = tensorflow.train.get_checkpoint_state(os.path.dirname(configuration.CHECKPOINT_PATH + "/checkpoint"))
    if checkpoint and checkpoint.model_checkpoint_path:
        print("Loading...")
        save_instance.restore(s, checkpoint.model_checkpoint_path)
    else:
        print("New Instance being created...")

def train():
    # Work on this!!!
    pass

def take_test(s, model, th):
    for hid in range(len(configuration.DATA_HOLDER)):
        if len(th[hid]) == 0:
            continue
        start = time.time()
        ei, di, dm = datamanagement.gb(
            th[hid],
            hid,
            bs = configuration.SIZE_PER_BATCH
        )
        _, s_loss, _ = do_training_step(s, model, ei, di, dm, hid, True)
        print("Test data holder {}: L = {}, T = {}s".format(hid, s_loss, time.time() - start))