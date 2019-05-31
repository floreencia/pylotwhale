#!/usr/bin/env python

from distutils.core import setup

setup(name='pylotwhale',
      version='1.0',
      description='bioacoustics related scripts',
      author='Florencia Noriega',
      author_email='flo@nld.ds.mpg.de',
      packages=['pylotwhale',
                'pylotwhale.NLP',
                'pylotwhale.signalProcessing', 
                'pylotwhale.utils', 
                'pylotwhale.MLwhales',
<<<<<<< HEAD
              ]
=======
               # 'pylotwhale.sequences'
               ]
>>>>>>> 7a093d0d465a900a819d0e5bf5c3850c1276cdb4
     )
