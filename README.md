# PylotWhale

[PylotWhale](https://github.com/floreencia/pylotwhale) is a Python package for automatically annotating bioacoustic recordings by combining available tools for machine learning (Scikit-learn) and audio signal processing (Librosa). With PylotWhale you can handle annotated audio files, extract audio features and transform classifier predictions into annotation files. 

## How does it work?

The framework relays on text annotations for audio — a standard and flexible format, handleable by many audio processing platforms. The classifier is trained with annotated recordings and its output is text annotations. The package supports two classification modes. In the first mode, classification instances are time frames of equal length from a recording. This classification scheme is suitable for detecting or segmenting animal sounds for example calls from whales or birds. In the second mode, the classification instances are recording segments of varying durations. This classification scheme is capable of classifying different types of vocal units, such as types of whale or bird calls. 

The modularity of the package enables us to easily tune the feature extraction procedure and type of classifier to suit the classification task at hand.

Have a look at the [demo notebook](http://nbviewer.jupyter.org/github/floreencia/pylotwhale/blob/master/examples/segment_Bat_B.ipynb) to get started.

Keywords: automatic annotation, classification, whale call, bat calls, python package, segmentation


## Dependencies

If you are using `pip`, you can run `pip install -r requirements.txt`,
and for `conda`, run `conda install --yes --file requirements.txt`
using the proper interpreter or environment.
