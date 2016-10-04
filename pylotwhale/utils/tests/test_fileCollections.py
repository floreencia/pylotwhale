# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 16:41:04 2015
@author: florencia
"""
from __future__ import print_function
import pylotwhale.utils.fileCollections as fcll

import os
import glob

#### File loading functions
sripts_dir = os.path.dirname(os.path.realpath(__file__))
teFi = sripts_dir+'/text.txt'

def test_get_path():
    return fcll.get_path(teFi)


def test_concatFile_intoList():
    return 2*len(fcll.concatFile_intoList(teFi)) == len(fcll.concatFile_intoList(teFi, teFi))
       
