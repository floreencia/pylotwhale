# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 17:45:10 2015
@author: florencia
"""

from __future__ import print_function
import numpy as np
from pylotwhale.NLP.ngramO_beta import *

m_test = np.random.randint(1, 10, (5, 3))
columns_test = list('xyz') # samples
rows_test = list('abcde') # conditions

df_test = ngr.matrix2DataFrame(m_test, rows=rows_test, columns=columns_test)



def test_matrix2DataFrame():
    df = ngr.matrix2DataFrame(m_test, rows=rows_test, columns=columns_test)
    assert df.loc['a', 'x'] == m[0,0]
    assert df.loc['b', 'y'] == m[1,1]


def test_DataFrame2kykyDict():
    kkD = ngr.DataFrame2kykyDict(df_test)
    assert thisdf.loc['a', 'x'] == kkD['a']['x']
