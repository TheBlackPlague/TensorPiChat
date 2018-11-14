# TensorPiChat
# A basic chat-bot using sequence-based learning using TensorFlow. Datastructure used is "Cornell's Movie Dialog Corpus".
# Licensed under MIT - Developed by Shaheryar Sohail. By the MIT License, you can do whatever with any code from this file as long as you agree to have provide a commented in citation of original source.

import tensorflow
import time
import numpy
import configuration

class TensorPiDM:

    def graph(self):
        self.placeholder_creation()
        self.inference()
        self.make_loss()
        self.optimize_machine()
        self.summary()

    def __init__(self, forward_only, bs):
        print("New DataModel! Setting up...")
        self.fwo = forward_only
        self.bs = bs

    def placeholder_creation(self):
        self.ei = [
            tensorflow.placeholder(tensorflow.int32, shape = [None], name = "encoder{}".format(i))
            for i in range(configuration.DATA_HOLDER[-1][0])
        ]
        self.di = [
            tensorflow.placeholder(tensorflow.int32, shape = [None], name = "decoder{}".format(i))
            for i in range(configuration.DATA_HOLDER[-1][1])
        ]
        self.dm = [
            tensorflow.placeholder(tensorflow.float32, shape = [None], name = "mask{}".format(i))
            for i in range(configuration.DATA_HOLDER[-1][1] + 1)
        ]
        self.t = self.di[1:]

    def inference(self):
        if configuration.SAMPLE_NUM > 0 and configuration.SAMPLE_NUM < configuration.DEC_DICT:
            w = tensorflow.get_variable("proj_w", [
                configuration.HIDDEN_LAYER_NUM, configuration.DEC_DICT
            ])
            b = tensorflow.get_variable("proj_b", [configuration.DEC_DICT])
            self.outputproj = (w, b)
        def loss_per_sample(ldigit, label):
            label = tensorflow.reshape(label, [-1, 1])
            return tensorflow.nn.sampled_softmax_loss(
                weights = tensorflow.transpose(w),
                biases = b,
                inputs = ldigit,
                labels = label,
                num_sampled = configuration.SAMPLE_NUM,
                num_classes = configuration.DEC_DICT
            )
        self.softmax_lossfx = loss_per_sample
        scell = tensorflow.contrib.rnn.GRUCell(configuration.HIDDEN_LAYER_NUM)
        self.cell = tensorflow.contrib.rnn.MultiRNNCell([
            scell for _ in range(configuration.INPUT_LAYER_NUM)
        ])
    
    def summary(self):
        # meh.. will do this some other time.
        pass

    def optimize_machine(self):
        print("This may take a bit of a while. Please stay patient.")
        with tensorflow.variable_scope("training") as scope:
            self.global_step = tensorflow.Variable(
                0,
                dtype = tensorflow.int32,
                trainable = False,
                name = "global_step"
            )
            if not self.fwo:
                self.optimizing_machine = tensorflow.train.GradientDescentOptimizer(configuration.LR)
                trainable_p = tensorflow.trainable_variables()
                self.gradient_normal = []
                self.train_operative = []
                start = time.time()
                for d_holder in range(len(configuration.DATA_HOLDER)):
                    clipped_gradient, normal = tensorflow.clip_by_global_norm(tensorflow.gradients(
                        self.loss[d_holder],
                        trainable_p
                    ), configuration.NORMALIZER)
                    self.gradient_normal.append(normal)
                    self.train_operative.append(self.optimizing_machine.apply_gradients(zip(clipped_gradient, trainable_p),
                        global_step = self.global_step
                    ))
                    print("Optimizer built for Data Holder {} and it took {}s".format(d_holder, time.time() - start))
                    start = time.time()

    def make_loss(self):
        print("This may take a bit of a while. Please stay patient.")
        start = time.time()
        def sequence_to_sequence_feed(ei, di, should_decode):
            setattr(tensorflow.contrib.rnn.GRUCell, "__deepcopy__", lambda self, _: self)
            setattr(tensorflow.contrib.rnn.MultiRNNCell, "__deepcopy__", lambda self, _:self)
            return tensorflow.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                ei, di, self.cell,
                esymbol_num = configuration.ENC_DICT,
                dsymbol_numb = configuration.DEC_DICT,
                embedding_size = configuration.HIDDEN_LAYER_NUM,
                outputproj = self.outputproj,
                f_prev = should_decode
            )
        if self.fwo:
            self.output, self.loss = tensorflow.contrib.legacy_seq2seq.model_with_buckets(
                self.ei,
                self.di,
                self.t,
                self.dm,
                configuration.DATA_HOLDER,
                lambda x, y: sequence_to_sequence_feed(x, y, True),
                softmax_lossfx = self.softmax_lossfx
            )
            if self.outputproj:
                for d_holder in range(len(configuration.DATA_HOLDER)):
                    self.output[d_holder] = [
                        tensorflow.matmul(output,
                        self.outputproj[0]) + self.outputproj[1] for output in self.output[d_holder]
                    ]
        else:
            self.output, self.loss = tensorflow.contrib.legacy_seq2seq.model_with_buckets(
                self.ei,
                self.di,
                self.t,
                self.dm,
                configuration.DATA_HOLDER,
                lambda x, y: sequence_to_sequence_feed(x, y, False),
                softmax_lossfx = self.softmax_lossfx
            )
        print("Done! It took {}s".format(time.time() - start))