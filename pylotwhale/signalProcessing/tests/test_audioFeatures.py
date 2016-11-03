#!/usr/bin/python

from __future__ import print_function, division
from py.test import raises
import numpy as np
import pylotwhale.signalProcessing.audioFeatures as auf


def test_texturiseWalkig():
	## nTextWS=1
	M = np.random.randint(5, 11, (20,20))
	Mtext = auf.texturiseWalkig(M, 1, normalise=False)
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
