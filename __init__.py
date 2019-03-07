import platform
if platform.python_version_tuple()[0] != '3':
    raise Exception("Please use python3 ^^")
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import SymLogNorm, NoNorm
import numpy as np
import pandas as pd
import seaborn as sns
import os

"""
I finally want to work out common plots between all my scripts.
Common things that I want to use are 
- pdf/cdf/ccdf
"""

def pjoin(listo):
    return os.path.sep.join(listo)

def convert_to_time(string):
    return pd.to_datetime(string, format="%Y-%m-%d %H:%M:%S.%f")

def write_plot(figure, name):
    ftype = "svg"
    if "PFTYPE" in os.env:
        ftype = os.env["PFTYPE"]
    fpath = pjoin([os.curdir, "plot", "{}.{}".format(name, ftype)])
    print("Writing plot '{}'".format(fpath))
    figure.savefig(fpath, bbox_inches='tight')

def read_df(name):
    fpath = pjoin([os.curdir, "data", "{}.pickle".format(name)])
    print("Loading dataframe '{}'".format(fpath))
    return pd.read_pickle(fpath)

def write_df(df, name):
    fpath = pjoin([os.curdir, "data", "{}.pickle".format(name)])
    print("Writing dataframe '{}'".format(fpath))
    df.to_pickle(fpath)

def pdf(data, col, count_col=None):
    if '__count' in data.columns:
        raise Exception("__count already exists")
    if not count_col:
        data['__count'] = 1
        total = data['__count'].sum()
        res = data.groupby(col)['__count']
        res = res.sum() / total
        del data['__count']
    else:
        total = data[count_col].sum()
        res = data.groupby(col)[count_col]
        res = res.sum() / total
    return res

def cdf(data, col, count_col=None):
    if '__count' in data.columns:
        raise Exception("__count already exists")
    if not count_col:
        data['__count'] = 1
        total = data['__count'].sum()
        res = data.groupby(col)['__count']
        res = res.sum() / total
        del data['__count']
    else:
        total = data[count_col].sum()
        res = data.groupby(col)[count_col]
        res = res.sum() / total
    res = res.cumsum()
    return res

def plotme(series, ax=None, ylog=True, xlog=False, loglog=False,
        linestyle="-", marker="", **kwargs):
    f = None
    if not ax:
        f, ax = plt.subplots()
    series.plot(ax=ax, linestyle=linestyle, marker=marker, **kwargs)
    if ylog or loglog:
        ax.set_yscale('symlog')
    if xlog or loglog:
        ax.set_xscale('symlog')
    #ax.set_xlabel(col.lower().replace('_', ' '))
    ax.set_ylabel("probability")
    ax.set_title("PDF")
    if f:
        f.tight_layout(pad = 0)
        f.patch.set_visible(False)
    return ax

def plot_pdf(data, col, *args, **kwargs):
    res = pdf(data, col)
    ax = plotme(res, *args, marker="o", **kwargs)
    ax.set_title("PDF")
    return ax

def plot_cdf(data, col, *args, **kwargs):
    res = cdf(data, col)
    ax = plotme(res, *args, **kwargs)
    ax.set_title("ECDF")
    return ax

def plot_multi_cdf(data, cols, *args, **kwargs):
    ax = None
    for col in cols:
        d = cdf(data, col)
        ax = plotme(d, *args, ax=ax, label=col, **kwargs)
    lines = ax.get_lines()# + ax.right_ax.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc='lower right')
    ax.set_xlabel("variable")
    ax.set_title("CDF")
    return ax

def plot_ccdf(data, col, *args, **kwargs):
    res = 1-cdf(data, col)
    ax = plotme(res, *args, **kwargs)
    ax.set_title("CCDF")
    return ax

def top(data, key_col, value_col, n=10):
    return data.sort_values(value_col, ascending=False).reset_index(drop=True).head(n)

def plot_top(data, key_col, value_col, n=10):
    res = top(data, key_col, value_col, n=n)
    ax = res[value_col].plot()
    fig.tight_layout(pad = 0)
    fig.patch.set_visible(False)
    return ax

def heatmap_test_data(ncols=3, nrows=10, log=False):
    """ Create test data for heatmap with keys 'time' and 'variable'
    """
    if log:
        arr = np.power(10, np.random.rand(nrows, ncols)*10)
    else:
        arr = np.random.randint(1, 1000, (nrows, ncols))
    d = pd.DataFrame(arr)
    d['time'] = pd.date_range("1/1/2018", periods=nrows, freq='H')
    return pd.melt(d, id_vars="time")

def heatmap(data, key1, key2, values='value', xlabel="", ylabel="", log=False,
        carryover=False):
    """ Plot a heatmap with barcharts on the side
        key1, key2 - keys of the x and y dimension
    """
    def disable_ticks(ax):
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())

    def disable_axes(ax):
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    #cmap = "YlGnBu"
    cmap = "CMRmap"
    bar_col = "silver"

    fig = plt.figure(frameon=False)

    # create a 2 x 3 subplot field
    gs = gridspec.GridSpec(2, 3, width_ratios=[5, 1, 1], height_ratios = [1, 5])
    ax_center = plt.subplot(gs[1,0], frameon=False)
    ax_top = plt.subplot(gs[0,0], frameon=False, sharex=ax_center)
    ax_right = plt.subplot(gs[1,1], frameon=False)
    ax_right2 = plt.subplot(gs[1,2], frameon=False, sharex=ax_center)

    # prepare data
    matrix = data.pivot(key1, key2, values=values)
    if carryover:
        matrix = matrix.fillna(method='ffill')
    else:
        matrix = matrix.fillna(0)
    print("matrix.shape={}".format(matrix.shape))
    top = data.groupby(key1)[values].sum()
    right = data.groupby(key2)[values].sum()[::-1]

    # plot
    hm_data = matrix.T
    if log:
        norm = SymLogNorm(vmin=hm_data.min().min(), vmax=hm_data.max().max(),
                linthresh=1)
        hm = sns.heatmap(hm_data, ax=ax_center, cbar=False, cmap=cmap,
                norm=norm, linewidths=0)
    else:
        hm = sns.heatmap(hm_data, ax=ax_center, cbar=False, cmap=cmap, linewidths=0)
    # previously used imshow, didn't work as nicely
    ## ax_center.imshow(matrix.T, aspect='auto')
    top.plot.bar(ax=ax_top, color=bar_col, align='edge', log=log)
    #ax_top.set_yscale('symlog')
    right.plot.barh(ax=ax_right, color=bar_col, align='edge', log=log)
    #ax_top.set_yscale('symlog')
    plt.colorbar(ax_center.get_children()[0], ax=ax_right2, orientation='vertical')

    # naming
    ax_center.set_xlabel(xlabel)
    ax_center.set_ylabel(ylabel)

    # remove axes and ticks where possible
    disable_axes(ax_right2)
    disable_axes(ax_top)
    disable_axes(ax_right)
    disable_ticks(ax_center)

    # remove spaces
    fig.tight_layout(pad = 0)
    fig.patch.set_visible(False)
    return fig, [ax_top, ax_right, ax_right2, ax_center], {key1:top, key2:right}

def plot_heatmap(*args, **kwargs):
    return heatmap(*args, **kwargs)
