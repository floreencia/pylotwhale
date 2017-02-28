# pylotwhale

Scripts for anaylising bioacoustic signals

train and save classifier with:
/MLwhales/chooseFeatures.py

uses:

* scikit-learn 0.17. 
* matplotlib 1.3.1. 
* numpy 1.8.2.
* librosa (https://github.com/bmcfee/librosa) 
* http://python-speech-features.readthedocs.org/en/latest/


Toolboxes
=========

Code is organised into the following modules:

utils
-----

Plotting handling text, paths, collections, general to diverse contexts, not only ML or bioacoustics.

#### whaleFileProcessing
* file name processing, time stamp interpreting
* code
* parses Heikes file convention
* → parse orchive file 
convention

#### fileCollections
* functions used in the creation of collections
* aupTextFile2mtlFile functions

#### annotationsTools
* functions used in the creation and parsing of annotation files
* some aupTextFile2mtlFile functions are here

#### dataTools
* tools for handling data: collections, filtering, grouping

#### netTools
* tools for creating plots with networkx and graphviz

#### plotTools
* plotting tools


SignalProcessing
----------------
diverse tools for working with signals (1dim time series)

#### signaltools_beta
code
* visualization
* feature extraction
	* cepstral, spectral, delta, etc
* feature texturization
* annotations handling
	* tuple ts → sections dict

#### audioFeatures
* ... under construction! link between the audio representations and fex
	* summarisation, etc

#### effects
* audio effects: add white noise, pitch shift, time streching

MLwhales
--------

diverse tools for supervisedly detecting and classifying sounds using wave files and annotations.

#### MLtools_beta
code
* preprocessing
	* removing buggy instances/features
* visualizing features (annotated/and not ann)
* clf scores
* objects for feature data X,y

#### featureExtraction
code
* functions
	* readcols - read columns
* classes
	* feature extraction, for spiting and walking

#### prediction tools
code
* plot confusion matrix
* predict and write

#### experimentTools
* tools running 
classification experiments under diverse conditions

NLP
---

Tools for analysing annotations

#### annotations_analyzer
code
* annotations object
	* ict
	* call lengths
	* plot tape
* plotAnnotatedSpectro
	* visualise annotations plotting them together / the spectrogram

#### sequencesO_beta
code
* csv → df → bigrams dict
* plot tape

#### myStatistics_beta
statistical tests: difference of proportions, chi-square
