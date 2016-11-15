from __future__ import print_function, division
from py.test import raises
import numpy as np
import pylotwhale.MLwhales.MLtools_beta as myML
#import pylotwhale.signalProcessing.audioFeatures as auf


def test_dataXy_filter():
    # test dataXy loading data
    M = np.random.randint(1, 5, (4,4))
    labs = np.random.randint(0,1, (4,))
    datO = myML.dataXy_names(M, labs)
    np.testing.assert_array_equal(M, datO.X)

    # test None filter form the data_ynames class
    M_NoneFilt, labs_NoneFilt = datO.filterInstances(None)
    np.testing.assert_array_equal(M, M_NoneFilt) # filtering


