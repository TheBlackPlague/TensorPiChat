# TensorPiChat
# A basic chat-bot using sequence-based learning using TensorFlow. Datastructure used is "Cornell's Movie Dialog Corpus".
# Licensed under MIT - Developed by Shaheryar Sohail. By the MIT License, you can do whatever with any code from this file as long as you agree to have provide a commented in citation of original source.

import random
import os
import numpy
import configuration
import re

if __name__ == "__main__":
    prepare_d()
    process_d()

def make_directory(dir_path):
    # Checks if directory exists and makes it if directory doesn't exist.
    try:
        os.mkdir(dir_path)
    except OSError:
        pass

def t_generator(line, n_d = True):
    # Thanks to Tensorflow Sample code - Go Google I guess?
    line = re.sub("<u>", "", line)
    line = re.sub("</u>", "", line)
    line = re.sub("\[", "", line)
    line = re.sub("\]", "", line)
    w = []
    SPLIT_W = re.compile("([.,!?\"'-<>:;)(])")
    RE_DIGIT = re.compile(r"\d")
    for f in line.strip().lower().split():
        for t in re.split(SPLIT_W, f):
            if not t:
                continue
            if n_d:
                t = re.sub(RE_DIGIT, "#", t)
            w.append(t)
    return w

def s2i(v, l):
    return [
        v.get(t, v['<unk>']) for t in t_generator(l) 
    ]

def t2i(d, m):
    v_p = "vocab." + m
    i_p = d + "." + m
    o_p = d + "id." + m
    _, dict = read_dictionary(os.path.join(configuration.UNKNOWN_DATA_PATH, v_p))
    i_f = open(os.path.join(configuration.UNKNOWN_DATA_PATH, i_p), "r")
    o_f = open(os.path.join(configuration.UNKNOWN_DATA_PATH, o_p), "w")
    line = i_f.read().splitlines()
    for l in line:
        if m == "dec":
            id = [dict["<s>"]]
        else:
            id = []
        id.extend(s2i(dict, l))
        if m == "dec":
            id.append(dict["<\s>"])
        o_f.write(" ".join(str(i) for i in id) + "\n")

def data_set_init(question, answer):
    make_directory(configuration.UNKNOWN_DATA_PATH)
    # Once directory has been made and sorted out lets begin the preparing dataset.
    tid = random.sample([i for i in range(len(question))], configuration.MAX_SIZE_PER_BATCH)
    name = ["train.enc", "train.dec", "tid.enc", "tid.dec"]
    file = []
    for n in name:
        joinedpath = os.path.join(configuration.UNKNOWN_DATA_PATH, n)
        file.append(open(joinedpath, "w"))
    # After appending
    for i in range(len(question)):
        if i in tid:
            file[2].write(question[i] + "\n")
            file[3].write(answer[i] + "\n")
        else:
            file[0].write(question[i] + "\n")
            file[1].write(answer[i] + "\n")
    # After checking in range.        
    for f in file:
        f.close()

def div_data_set(i2l, convo):
    # Divide dataset into 2 mediums : Questions / Answers
    question = []
    answer = []
    for c in convo:
        for i, l in enumerate(c[:-1]):
            question.append(i2l[i])
            answer.append(i2l[i + 1])
    assert len(question) == len(answer)
    return question, answer

def make_dictionary(file, n_d = True):
    i_p = os.path.join(configuration.UNKNOWN_DATA_PATH, file)
    o_p = os.path.join(configuration.UNKNOWN_DATA_PATH, "vocab.{}".format(file[-3:]))
    vocab = {}
    with open(i_p, "r") as f:
        for l in f.readlines():
            for t in t_generator(l):
                if not t in vocab:
                    vocab[t] = 0
                vocab[t] += 1
    s_vocab = sorted(vocab, key = vocab.get, reverse = True)
    with open(o_p, "w") as f:
        f.write("<pad>" + "\n")
        f.write("<unk>" + "\n")
        f.write("<s>" + "\n")
        f.write("<\s>" + "\n")
        i = 4
        for w in s_vocab:
            if vocab[w] < configuration.THRESHOLD_PER_BATCH:
                break
            f.write(w + "\n")
            i += 1
        with open("configuration.py", "a") as secondf:
            if file[-3:] == "enc":
                secondf.write("ENC_VOCAB = " + str(i) + "\n")
            else:
                secondf.write("DEC_VOCAB = " + str(i) + "\n")
        
def read_dictionary(d_path):
    with open(d_path, "r") as f:
        w = f.read().splitlines()
    return w, {
        w[i]: i for i in range(len(w))
    }

def gl():
    i2l = {}
    f_p = os.path.join(configuration.KNOWN_DATA_PATH, configuration.MOVIE_LINE)
    print(configuration.MOVIE_LINE)
    with open(f_p, "r", errors = "ignore") as f:
        i = 0
        try:
            for l in f:
                p = l.split(" +++$+++ ")
                if len(p) == 5:
                    if p[4][-1] == "\n":
                        p[4] = p[4][:-1]
                    i2l[p[0]] = p[4]
                i += 1
        except UnicodeDecodeError:
            print(i, l)
    return i2l

def gc():
    f_p = os.path.join(configuration.KNOWN_DATA_PATH, configuration.CONVERSATION)
    c = []
    with open(f_p, "r") as f:
        for l in f.readlines():
            p = l.split(" +++$+++ ")
            if len(p) == 4:
                c_prime = []
                for l in p[3][1:-2].split(", "):
                    c_prime.append(l[1:-1])
                c.append(c_prime)
    return c

def prepare_d():
    print("Training using raw data...")
    i2l = gl()
    c = gc()
    q, a = div_data_set(i2l, c)
    data_set_init(q, a)
    print("Done!")

def process_d():
    print("Processing data into model...")
    make_dictionary("train.enc")
    make_dictionary("train.dec")
    t2i("train", "enc")
    t2i("train", "dec")
    t2i("tid", "enc")
    t2i("tid", "dec")

def load_d(enc, dec, mts = None):
    e_f = open(os.path.join(configuration.UNKNOWN_DATA_PATH, enc), "r")
    d_f = open(os.path.join(configuration.UNKNOWN_DATA_PATH, dec), "r")
    e = e_f.readline()
    d = d_f.readline()
    d_holder = [[] for _ in configuration.DATA_HOLDER]
    i = 0
    while e and d:
        if (i + 1) % 10000 == 0:
            print("One double value", i)
        e_id = [int(id) for id in e.split()]
        d_id = [int(id) for id in d.split()]
        for h_id, (e_ms, d_ms) in enumerate(configuration.DATA_HOLDER):
            if len(e_id) <= e_ms and len(d_id) <= d_ms:
                d_holder[h_id].append([e_id, d_id])
                break
        e = e_f.readline()
        d = d_f.readline()
        i += 1
    return d_holder

def input_based_on_pad(i, s):
    return i + [configuration.PAD_ID] * (s - len(i))

def rebatch(i, s, bs):
    b_input = []
    for lid in range(s):
        b_input.append(numpy.array([
            i[bid][lid] for bid in range(bs)
        ], dtype = numpy.int32))
    return b_input

def gb(d_holder, h_id, bs = 1):
    # Thanks Stackoverflow! You fucking geniuses on there!
    es = configuration.DATA_HOLDER[h_id]
    ds = configuration.DATA_HOLDER[h_id]
    ei = []
    di = []
    for _ in range(bs):
        ei, di = random.choice(d_holder)
        ei.append(list(reversed(input_based_on_pad(ei, es))))
        di.append(input_based_on_pad(di, ds))
    bei = rebatch(ei, es, bs)
    bdi = rebatch(di, ds, bs)
    bm = []
    for lid in range(ds):
        bm = numpy.ones(bs, dtype = numpy.float32)
        for bid in range(bs):
            if lid < ds - 1:
                t = di[bid][lid + 1]
            if lid == ds - 1 or t == configuration.PAD_ID:
                bm[bid] = 0.0
        bm.append(bm)
    return bei, bdi, bm

