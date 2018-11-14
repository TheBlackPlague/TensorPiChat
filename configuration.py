# TensorPiChat
# A basic chat-bot using sequence-based learning using TensorFlow. Datastructure used is "Cornell's Movie Dialog Corpus".
# Licensed under MIT - Developed by Shaheryar Sohail. By the MIT License, you can do whatever with any code from this file as long as you agree to have provide a commented in citation of original source.

KNOWN_DATA_PATH = "knowndata"
UNKNOWN_DATA_PATH = "unknowndata"
INPUT_LAYER_NUM = 3
HIDDEN_LAYER_NUM = 256
SIZE_PER_BATCH = 64
THRESHOLD_PER_BATCH = 2
MAX_SIZE_PER_BATCH = 25000
# The learning Rate.
LR = 0.5
# Sample number based on Hidden Layer Number. Since LR is half (1/2), this should be double Hidden Layer -> 256 * 2 = 512.
SAMPLE_NUM = 512
# Normalizer to get values out of 1. Based on percentages. Ten times of LR.
NORMALIZER = 5
DATA_HOLDER = [
    (19, 19),
    (28, 28),
    (33, 33),
    # incremented rate on data buffers.
    (40, 43),
    (50, 53),
    (60, 63)
]

# ID (s) saved by Index
PAD_ID = 0
UNK_ID = 1
START_ID = 2
EOS_ID = 3

# Output & Real-time processed stack
CHECKPOINT_PATH = "output/checkpoint"
OUTPUT_FILE_PATH = "ouput/file.txt"
MOVIE_LINE = "output/line/file.txt"
CONVERSATION = "output/convo/file.txt"

# English data rules
GRAMMER_CONTRACTION = [
    ("don ' t ", "do n't "),
    ("didn ' t ", "did n't "),
    ("can ' t ", "ca n't "),
    ("doesn ' t ", "does n't "),
    ("wouldn ' t ", "would n't "),
    ("shouldn ' t ", "should n't "),
    ("i ' m ", "i 'm "),
    ("' d ", "'d "),
    ("in ' ", "in' "),
    ("' s ", "'s "),
    ("' ve ", "'ve "),
    ("' re", "'re ")
]
ENC_DICT = 24414
DEC_DICT = 24691