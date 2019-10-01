from __future__ import print_function
import os
import glob
import warnings
from collections import Counter

"""
Module for creating collections and handling paths of annotation files

Some features are:
- finding paths in directories
-check is the items of a collection are repeated
- creating collections by searching files in a directory

A collection is a text file with paths to particular files,
eg. audio files or annotation files

"""

#### File loading functions


def get_path(fiPath):  # checks if file exists
    return os.path.isfile(fiPath)


def concatFile_intoList(*path2files):  # load text files and concat lines into list
    lines = []
    for fi in path2files:
        with open(fi, "r") as f:
            lines.extend(f.read().strip().splitlines())
    return lines


### Check for duplicates


def areListItemsUnique(mylist):
    """checks for duplicated items in a list, if False,
    elements and their indexes are printed"""
    uniqueItems = set(mylist)
    uniqueLi = list(uniqueItems)
    if len(mylist) == len(uniqueLi):  # no duplicates
        return True
    else:  # contains duplicates
        li_counts = Counter(mylist)
        collStr = ""
        for i in range(len(mylist)):
            if li_counts[mylist[i]] > 1:
                collStr += "\n ---> {} {}".format(i, mylist[i])
        print(collStr)
        return False


def getUniqueItemIndexes(mylist):
    """returns the indexes of the set of elements in the list (mulist),
    i.e. excluding duplicates"""
    uniqueLi = list(set(mylist))
    if len(mylist) != len(uniqueLi):
        warnings.warn("WARNING! Duplicated items")
        # warnings.warn("Duplicated items")
        return sorted([mylist.index(item) for item in uniqueLi])
    else:  # no dup indexes
        # "unique items list")
        return range(len(mylist))  # return all indexes


def filterListForUniqueItems(myList):
    uIdx = getUniqueItemIndexes(myList)
    return [myList[item] for item in uIdx]


def areLinesUnique(*collFiles):
    li = concatFile_intoList(*collFiles)
    if areListItemsUnique(li):
        print("no duplicates")
        return True
    else:
        print("contains duplicates")
        return False


### collection creating


def annotationsDir2wavAnnCollection(
    annDir, wavDir="default", outCollFile="default", str0=".txt", strRep=".wav"
):
    """
    reads the annotation files (*.txt) in annDir,
    searches their corresponding waves in
    wavDir and saves a collection <outCollFile> of annotated wavs
    """
    if wavDir == "default":
        wavDir = os.path.join(annDir, "..")
    if outCollFile == "default":
        outCollFile = os.path.join(annDir, "..", "collection.txt")

    annotationsFiList = glob.glob(os.path.join(annDir, "*.txt"))  # enlist all the annotations
    return annotationsList2wavAnnCollection(
        annotationsFiList, wavDir, outCollFile, str0=str0, strRep=strRep
    )


def annotationsList2wavAnnCollection(
    annotationsFiList, wavDir, outCollFile, str0=".txt", strRep=".wav"
):
    """
    searches the wave files in the dir <wavDir>
    from each annotationFile in annotationsFiList
    and saves a collection <outCollFile> of annotated wavs
    Parameters
    ----------
    annotationsFiList : list of annotation files
    wavDir : directory where to find the waves
    outCollFile : out file name
    str0 : string of the template file to be replaced
    strRep : characteristic string of the file we are looking for
    """
    with open(outCollFile, "w") as g:
        for annFi in annotationsFiList:
            wavFileName = os.path.basename(annFi).replace(str0, strRep)
            # print(wavFileName)
            wavFiPath = findFilePathInDir(wavFileName, wavDir)
            if wavFiPath:
                g.write("{}\t{}\n".format(wavFiPath, annFi))
            else:
                print(wavFileName, "not found")
    return outCollFile


def findFilePathInDir(guessFile, searchDir):
    """
    looks for the path of a file (base name),
    assuming this is in searchDir
    """
    for root, dirs, files in os.walk(searchDir):
        for name in files:
            if name == guessFile:
                return os.path.abspath(os.path.join(root, name))
    return False


def findFilesInThreeDir(searchDir, startFN="", endFN=""):
    """searches for all the files with a given
    end and or ending in a directory all the subdirectories
    returns a list with the filenames"""
    filesList = []
    for root, dirs, files in os.walk(searchDir):
        for f in files:
            if f.endswith(endFN):
                if f.startswith(startFN):
                    filesList.append(os.path.join(root, f))
    return filesList
