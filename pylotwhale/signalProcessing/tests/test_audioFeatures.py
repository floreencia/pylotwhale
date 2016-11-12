#!/usr/bin/python

from __future__ import print_function, division
from py.test import raises
import numpy as np
import pylotwhale.signalProcessing.audioFeatures as auf


def test_texturiseWalking():
	## nTextWS=1
	M = np.random.randint(5, 11, (20,20))
	Mtext = auf.texturiseWalking(M, 1, normalise=False)
	np.testing.assert_array_equal( M , Mtext[:, :np.shape(M)[1]])

	## nText=1, normalised


def test_texturiseSplitting():
	m = np.arange(12).reshape(4,3)

	## Nslices=1
	r1=auf.texturiseSplitting(m, 1, normalise=False)
	r2=np.hstack((m.mean(axis=0), m.std(axis=0)))[np.newaxis,:]
	np.testing.assert_equal(r1, r2)

	## Nslices=matrix size
	r1=auf.texturiseSplitting(m, len(m), normalise=False)
	r2=np.hstack((m, np.zeros(np.shape(m))))
	np.testing.assert_equal(r1, r2)


def test_time2indices():
    
    #### full interval ####
    tf_rec = np.random.rand() # rec length
    m = 102 # num of instances
    arr = np.linspace(0, tf_rec, m) # times array
    ## section times
    t0_s = 0
    tf_s = tf_rec
    ## get indices
    i_0, i_f = auf.time2indices(t0_s, tf_s, arr)
    assert arr[i_0] == t0_s
    assert arr[i_f] == tf_s
    assert i_0 == 0
    assert i_f+1 == m
    
    ### first fraction
    tf_rec = 20
    m = 100
    arr = np.linspace(0, tf_rec, m)
    
    n_frac = 5
    t0_s = 0
    tf_s = tf_rec/n_frac

    i_0, i_f = auf.time2indices(t0_s, tf_s, arr)

    assert( i_0 == 0)
    ### check end index
    # add one because index counting starts at 0
    # ceil for even numbers
    assert(i_f + 1 == int(np.ceil(m/n_frac))) 


def test_annotations2instanceArray():

    # particular case, labels hierarchy
    T = np.array([[1.,2.], [3.,4.], [6, 7], [6,8]])
    L = np.array(['c', 'c', 'c', 'x'])
    m = 10
    tf = 10.
    labsH = ['w','c', 'x']
    
    np.testing.assert_array_equal( auf.annotations2instanceArray(T, L, m, tf, labelsHierarchy=labsH), 
                                   np.array(['b', 'c', 'b', 'c', 'b', 'b', 'c', 'x', 'b', 'b']))

