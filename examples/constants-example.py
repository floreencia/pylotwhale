#!/usr/bin/env python
# encoding: utf-8
'''Update paths and rename as constants.py'''

import os

TEST = 1

BASE_PATH = '/path/to/pylotwhale/examples'
AUDIOS_DIR_PATH = '/path/to/AudioMNIST/data'

LABELLED_COLL = os.path.join(BASE_PATH, 'audios_collection.csv')

SPECTROS_DIR = os.path.join(BASE_PATH, 'figs')


### classifier params

# fft_ws = 512


