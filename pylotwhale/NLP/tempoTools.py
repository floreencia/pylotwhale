from __future__ import print_function, division

import numpy as np

from collections import Counter
from copy import deepcopy
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import entropy

import pylotwhale.utils.dataTools as daT


"""
tools for the preparation of annotated files
"""


def y_histogram(
    y,
    range=(0, 1.5),
    Nbins=None,
    oFig=None,
    figsize=None,
    plTitle=None,
    xl=r"$\tau _{ict}$ (s)",
    max_xticks=None,
):
    ## remove nans and infs
    y = y[~np.logical_or(np.isnan(y), np.isinf(y))]
    ## define number of bins
    if Nbins is None:
        Nbins = int(len(y) / 10)
    ## plot
    fig, ax = plt.subplots(figsize=figsize)
    # plt.figure(figsize=figsize)
    ax.hist(y, range=range, bins=Nbins)
    ax.set_xlabel(xl)  # in ({}, {}) s".format(rg[0], rg[1]))
    if isinstance(plTitle, str):  # title
        ax.set_title(plTitle)
    if isinstance(range, tuple):  # delimit plot
        ax.set_xlim(range)

    if max_xticks > 1:
        xloc = plt.MaxNLocator(max_xticks)
        ax.xaxis.set_major_locator(xloc)

    ## savefig
    if isinstance(oFig, str):
        fig.savefig(oFig, bbox_inches="tight")
    # print(oFig)
    return fig, ax


def pl_ic_bigram_times(
    df0,
    my_bigrams,
    ignoreKeys="default",
    label="call",
    oFig=None,
    violinInner="box",
    yrange="default",
    ylabel="time (s)",
    minNumBigrams=5,
):
    """violin plot of the ict of a my_bigrams
    Parameters:
    -----------
        df0 : pandas dataframe with ict column
        mu_bigrams : sequence to search for
        ignoreKeys : 'default' removes  ['_ini', '_end']
        label : type of sequence
        oFig : output figure
        violinInner : violin loc parameter
        yrange : 'default' (0, mu*2)
    """

    if ignoreKeys == "default":
        ignoreKeys = ["_ini", "_end"]

    topBigrams = daT.removeFromList(daT.returnSortingKeys(Counter(my_bigrams)), ignoreKeys)
    bigrTimes = []
    bigrNames = []

    for seq in topBigrams:
        df = daT.returnSequenceDf(df0, seq, label=label)
        # print(len(df))
        ict = df.ict.values
        if len(ict) > minNumBigrams:
            # bigrTimes[tuple(seq)] = ict[ ~ np.isnan(ict)]
            bigrTimes.append(ict[~np.isnan(ict)])
            bigrNames.append(seq)

    kys = ["{}{}".format(a, b) for a, b in bigrNames]
    sns.boxplot(bigrTimes, names=kys)

    if yrange == "default":
        meanVls = [np.mean(item) for item in bigrTimes if len(item) > 1]
        yrange = (0, np.mean(meanVls))
    plt.ylim(yrange)
    plt.ylabel(ylabel)
    plt.savefig(oFig, bbox_inches="tight")


def pl_calling_rate(
    df,
    t_interval=10,
    t0="t0",
    xL="time, [s]",
    yL=r"$\lambda$",
    max_xticks=None,
    plTitle=None,
    oFig=None,
):
    """plots the calling rate: # calls/t_interval for one tape dataframe"""
    call_t = df[t0].values
    ti = 0
    ## define time bins
    times_arr = np.arange(t_interval, call_t[-1] + t_interval, t_interval)
    ## initialise the calling rate for each time bin with zeros
    call_rate = np.zeros(len(times_arr))
    for (i, tf) in enumerate(times_arr):  # count the number of calls in each time bin
        call_rate[i] = len(call_t[np.logical_and(call_t > ti, call_t < tf)])
        ti = tf

    fig, ax = plt.subplots()
    ax.plot(times_arr, call_rate, marker="x")
    ax.set_xlabel(xL)
    ax.set_ylabel(yL)
    plt.autoscale()

    if max_xticks > 1:
        xloc = plt.MaxNLocator(max_xticks)
        ax.xaxis.set_major_locator(xloc)

    if plTitle:
        ax.set_title(plTitle)

    if oFig:
        fig.savefig(oFig, bbox_inches="tight")  # , bbox_inches='tight')


def pl_ictHist_coloured(ict, ict_di, bigrs, Nbins, rg=None, xL=r"$\tau _{ict}, [s]$", oFig=None):
    """
        Parameters:
        -----------
        ict: array, series
            all ict values, eg. df['ict_end_start']
        ict_di: dict
            ict by bigram
        bigrs: list
            with the bigram names,
            ngr.selectBigramsAround_dt(ict_di, (ixc_min, ixc_max), minBigr)
        rg: tuple
            hitogram range rg = (ixc_min, ixc_max)

        """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(ict, range=rg, label="other", alpha=0.4, color="gray", bins=Nbins)

    cmap = plt.cm.gist_rainbow(np.linspace(0, 1, len(bigrs)))
    ax.hist(
        [ict_di["".join(item)] for item in bigrs[:]],
        stacked=True,
        range=rg,
        label=bigrs,
        rwidth="stepfilled",
        bins=Nbins,
        color=cmap,
    )
    ax.set_xlabel(xL)
    plt.legend()

    if oFig:
        fig.savefig(oFig)

    return fig, ax


### ICIs


def check_finiteness(s1, s2):
    """filters non finite elements in either of both series"""
    assert len(s1) == len(s2)
    mask = np.logical_and(np.isfinite(s1), np.isfinite(s2))
    x = s1[mask]
    y = s2[mask]

    return x, y


def key_shuffle(d):
    """shuffles the values (array items)
    Parameters:
    -----------
    d: dict of numpy arrays
    """

    dsh = deepcopy(d)
    for k in d.keys():
        np.random.shuffle(dsh[k])
    return dsh


def flatten_dictValues(d1, d2):
    assert set(d1.keys()) == set(d2.keys())
    arr1 = []
    arr2 = []

    for k in d1.keys():
        arr1.extend(d1[k])
        arr2.extend(d2[k])
    return np.array(arr1), np.array(arr2)


def get_ici_i_series(df_dict, timeLabel="ict_end_start", i=1, fun=np.log, check_isfinite=True):
    """
    Parameters
    ----------
    df_dict: dictionary of pandas dataframe
    timeLabel: str
        name of the column to read
    Returns
    -------
    (s1, s2): 2D-tuple with pandas series
    """

    ict1, ict2 = get_ici_i(df_dict, timeLabel=timeLabel, i=i)
    ## apply fun
    ict1_log = fun(ict1)
    ict2_log = fun(ict2)
    ## filter non_infinites
    ict1_log, ict2_log = check_finiteness(ict1_log, ict2_log)

    s1 = pd.Series(ict1_log, name=r"$\log (\tau _i)$")
    s2 = pd.Series(ict2_log, name=r"$\log (\tau _{i+%d})$" % i)

    return s1, s2


def get_ici_i(df_dict, timeLabel="ict_end_start", i=1):
    """returns ici and ici_i as numpy arrays"""
    ict1_l = []
    ict2_l = []
    for tape in df_dict.keys():
        ict0 = df_dict[tape][timeLabel].values
        # ict = ict0[np.isfinite(ict0)]
        ict1_l.extend(ict0[i:])
        ict2_l.extend(ict0[:-i])

    return np.array(ict1_l), np.array(ict2_l)


def get_ici_i_DictSeries(df_dict, timeLabel="ict_end_start", i=1, fun=np.log, check_isfinite=True):
    """
    returns two dictionaries with the series of the timeLabel
        e.g. the ici-i calls away
    """
    s1 = {}
    s2 = {}
    for tape in df_dict.keys():
        ict0 = df_dict[tape][timeLabel].values
        # ict = ict0[np.isfinite(ict0)]
        x = ict0[i:]
        y = ict0[:-i]
        x, y = check_finiteness(x, y)
        s1[tape] = x
        s2[tape] = y

    return s1, s2


#### PLOTTING


def plTimeIntervals(
    datB,
    plTitle="",
    outFig="",
    shuffl=0,
    maxNTicks=False,
    TLims=(0, 19),
    frac=0,
    xLabel="ict [s]",
    yLabel="N",
):
    """
    DEPRECATE
    time interval distributions
    takes:
    < datB, data frame we whant to look into the time stamps
    < groupN, this is only given for the naming of the output plot
    < outFig, can be: outDir+"timeDistHist_%s.eps"%groupN
    * frac, display a line at the value of T below which frac*100%
            of the call pars are semarated, e.e, frac =0.75

    and returnsplots an histogram in the image folder

    """
    rec_datFrames, recs = sortedRecDatFr(datB, shuff=shuffl)  # recording data frames

    # time interval histogram
    interDist = {
        rec: rec_datFrames[rec].intervals.values for rec in rec_datFrames
    }  # intervals dictionary by recording

    # interval ditribution
    allT = list(interDist.values())  # all interval values
    allT = np.asarray([item for subli in allT for item in subli])  # flattern the intervals
    allT.reshape((len(allT), 1))  # reshape for dictionary
    allT = allT[~np.isnan(allT)]  # filter nans out
    print(np.shape(allT), allT.min(), allT.max())

    # histogram
    fig, ax = plt.subplots(figsize=(6, 3))
    cax = ax.hist(allT[~np.isnan(allT)], bins=allT.max(), color=([0, 0.49, 0.47]))
    ax.set_xlim(TLims)
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    if maxNTicks:
        plt.locator_params(nbins=maxNTicks)

    if frac:
        Cumx = np.cumsum(cax[0])
        Tot = sum(cax[0])
        tbins = cax[1]
        ax.axvline(x=sum(Cumx <= Tot * frac), linewidth=4, color=(1, 0.1, 0.0))  # [(0, 0.7, 0.9)])
        print("line at", sum(Cumx <= Tot * frac))

    if plTitle:
        ax.set_title(plTitle)

    if outFig:
        fig.savefig(outFig, bbox_inches="tight")
    print("outFig:", outFig)

    # return fig, ax


def plTape(t, yRaw, plName="", scaleF=20, lw=0.05, title=""):
    """
    plots the calls vs. time
    """
    assert len(t) == len(yRaw)

    freq_call = sorted(
        [(yRaw.count(ucall), ucall) for ucall in np.unique(yRaw)], reverse=True, key=lambda x: x[0]
    )  # sort calls
    i2c_tape = [thisCall[1] for thisCall in freq_call]
    c2i_tape = {i2c_tape[ix]: ix for ix in range(len(i2c_tape))}  # c2i
    # print np.unique(yRaw), c2i_tape, i
    # sys.exit()
    y = [c2i_tape[item] for item in yRaw]
    # print y[:5], i2c_tape, c2i_tape
    # plot

    # if not tapeN: tapeN = "%s%s%s"%(i2c_tape[0],len(i2c_tape), i2c_tape[-1])
    # figN = outDir+"tape_%s.pdf"%tapeN
    print(((t[-1] - t[0]) / scaleF, np.min([np.max([1, len(freq_call) / 2]), 3])))
    fig = pl.figure(figsize=((t[-1] - t[0]) / scaleF, np.min([np.max([1, len(freq_call) / 2]), 3])))
    ax = fig.add_subplot(111)
    pl.plot(t, y, marker="|", lw=lw, markeredgewidth=1.5)
    ax.set_ylim(-0.5, len(c2i_tape))  # +0.1)
    ax.set_xlim(t[0] - 5, t[-1] + 5)
    ax.set_yticks(np.arange(len(c2i_tape)))
    ax.set_yticklabels(i2c_tape, fontsize=8)
    ax.set_xlabel("time [s]")
    ax.set_title(title)
    if plName:
        pl.savefig(plName, bbox_inches="tight")


### FOURIER


def window_times(onset_times, t0, tf):
    """windows onset_times between t0 and tf
    Parameters
    ----------
    onset_times : array_like, shape (n_samples,)
        onset times, assumes onset times are sorted so that calling
        it window makes sense.
    t0 : float
        start of window
    tf : float
        end of window
    """
    assert tf > t0, "t0 must be smaller than tf"
    assert daT.isArraySorted(onset_times), "must be sorted"
    return onset_times[np.logical_and(onset_times >= t0, onset_times <= tf)]


def binary_time_series(onset_times, Dt=0.1):
    """converts onset times into a time-continuous binary array
    having the time stamp of the last element as length of the time vector
    Parameters
    ----------
    ev_times: ndarray
        onset times
    Dt: int
        time sampling interval, 1/sampling rate
    t_end: float
    Returns
    -------
    t_vec: ndarray
        times array
    IO: ndarray
        a binary array with ones at the onset positions
    Fs: float
        sampling rate
    """
    t_vec = np.arange(0, onset_times[-1] + Dt, Dt)
    IO = np.zeros_like(t_vec)

    for ct in onset_times:
        IO[np.argmin(np.abs(t_vec - ct))] = 1

    Fs = int(1.0 / Dt)  # int(len(t_vec)/t_vec[-1]) # =
    return t_vec, IO, Fs


def binarise_times_in_window(times, t0, tf, Dt=0.1):
    """Crates binary signal from onset times
    by windowing every Dt
    Make sure all provided parameters have the same units.

    Parameters
    ----------
    times : array_like, shape (n_samples,)
            onset times (if in seconds => Dt in Hz)
    t0 : float
        start time
    tf : float
        end time
    Dt : float
        sampling step, Fs = 1/Dt

    Returns
    ------
    t_vec : array_like, shape (n_samples,)
        times
    IO : array_like, shape (n_samples,)
        onsets
    """
    winL = tf - t0
    win_times = window_times(times, t0, tf) - t0
    t0_vec, IO_0, Fs = binary_time_series(win_times, Dt=Dt)
    t_vec = np.arange(0, tf - t0 + Dt, Dt)
    IO = np.zeros_like(t_vec)
    IO[: len(IO_0)] = IO_0
    return t_vec, IO


def KLdivergence(feature_arr):
    """KL-divergence matrix between all the elements of the feature_arr"""

    dist = np.zeros((len(feature_arr), len(feature_arr))) + np.nan

    for i in np.arange(len(feature_arr)):
        for j in np.arange(len(feature_arr)):  # np.arange(len(feature_arr)):
            dist[i, j] = entropy(feature_arr[i], feature_arr[j])
    return dist


### OTHER


def nPVI(IOI):
    """
    quantifies the temporal variability in speech rhythm [Grabe & Low, 2002]
    """
    n = len(IOI)
    diff = np.abs(np.diff(IOI))
    av = (IOI + np.roll(IOI, -1))[:-1] / 2
    return 100 / (n - 1) * np.sum(diff / av)
