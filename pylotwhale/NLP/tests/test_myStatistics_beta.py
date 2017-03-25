# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 17:45:10 2015
@author: florencia
"""

from __future__ import print_function
import numpy as np
from pylotwhale.NLP.myStatistics_beta import *


def test_binary_time_series():
    s = [['c', 'c', 'c', 'c', 'b', 'b']]
    assert repsPropotion_in_listOfSeqs(s) == (4,5)
    
