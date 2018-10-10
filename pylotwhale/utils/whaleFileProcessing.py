from __future__ import print_function
import re
import os

"""
Parse strings -- Heike's naming
"""

#### Heike's name convention parsing


def parseHeikesNameConv(fileN):
    '''parse Heikes file-naming
    Parameters
    ----------
    fileN : string with a file name using Heike's convention
        e.g. ...NPW-033-J-B-090713_f50-8_00_01_52...
    Returns
    -------
    di : dictionary with the info encoded in the file name
        call, date, group, quality,
            tape, timestamp, whale <-- This are often problematic!
    '''
    di = {}
    bN = os.path.basename(fileN)
    # parse all
    m=re.search(r'(N\wW)-(\w+)-(\w)-(\w)-([0-9]{5,6})_(f[0-9]*-[0-9]*)_([0-9]{2}_[0-9]{2}_[0-9]{2})', bN)

    try:  # to read parser groups
        di = {"whale": m.group(1), "call": m.group(2),
              "group": m.group(3), "quality": m.group(4), "date": m.group(5),
              "tape": m.group(6), "timestamp": m.group(7)}
        return di
    except AttributeError:
        pass
        #print("cannot parse time stamp (7) groups")

    try:  # second try, try to parse (5) groups, at least until the date    
        m = re.search(r'(N\wW)-(\w+)-(\w+)-(\w)-([0-9]{5,6})', bN)
        di = {"whale": m.group(1), "call": m.group(2),
              "group": m.group(3), "quality": m.group(4), "date": m.group(5)}
        return di
    except AttributeError:  # not even possible to parse until the date
           #print("ERROR: filename doesn't match Heike's convention!\n\t%s"%bN)
           #print("    eg. NPW-033-J-B-090713_f50-8_00_01_52...")
           #return False
        pass
    try:  # second try, try to parse (2) - whale group and call type
        m = re.search(r'(N\wW)-(\w+)', bN)
        di = {"whale": m.group(1), "call": m.group(2)}
        return di
    except:  # not even possible to parse call type
        # print("ERROR: filename doesn't match Heike's convention!\n\t%s"%bN)
        # print("    eg. NPW-033-J-B-090713_f50-8_00_01_52...")
        return None


def timeStamp2seconds(timeStamp):
    '''
    parses timestamps with the format hh_mm_ss
    and returns the timestampstamp in seconds
    '''
    try:
        h, m, s = timeStamp.split("_")
        return(int(float(h) * 60 * 60 + float(m) * 60 + float(h)))
    except:
        return None


def c_Label2HeikesCallLabel(annFile, labKey='call', repL='c'):
    '''annotation-files with a single section,
    preplace c with the call label in the file name'''
    label = parseHeikesNameConv(annFile)[labKey]

    with open(annFile, 'r') as f:
        content = f.read()

    with open(annFile, 'w') as g:
        g.write(content.replace('c', label))
