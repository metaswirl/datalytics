import platform
if platform.python_version_tuple()[0] != '3':
    raise Exception("Please use python3 ^^")
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import SymLogNorm, NoNorm

from PIL.PngImagePlugin import PngImageFile, PngInfo
import subprocess
import numpy as np
import matplotlib
import pandas as pd
import seaborn as sns
import os
import inspect
import traceback
from colorama import Fore, Style
from collections import namedtuple
import tempfile
import sys
import json
from collections import OrderedDict
from datetime import datetime as dt
import traceback
import xml.etree.ElementTree as ElementTree
import re

"""
I finally want to work out common plots between all my scripts.
Common things that I want to use are 
- pdf/cdf/ccdf
"""

HDF_NAMESPACE = '/data'

def eprint(msg):
    print(msg, file=sys.stderr)

def eprint_red(cls, msg):
    eprint(Style.DIM + "ERROR: " + msg + Style.RESET_ALL)

class ConstraintChecker:
    result_nt = namedtuple("result", ["success", "msg", "exception"])


    @classmethod
    def print_results(cls, results, print_ex=True):
        eprint("")
        eprint("=======================================")
        eprint(Style.DIM + "Constraints: " + Style.RESET_ALL)
        for result in results:
            eprint(result.msg)
            if result.exception and print_ex:
                print(result.exception)
        eprint("=======================================")
        eprint("")


    @classmethod
    def check_constraints(cls, constraints, verbose=False):
        results = []
        overall_status = True
        for c in constraints:
            msg = inspect.getsourcelines(c)[0][0]
            msg = msg.split('lambda: ')[1].strip()
            msg = msg[:-1] if msg.endswith(',') else msg
            status = False
            exception = None
            try:
                status = c()
            except Exception as ex:
                exception = repr(ex)
            if exception:
                msg = Fore.YELLOW + msg + Fore.RESET
            elif status:
                msg = Fore.GREEN + msg + Fore.RESET
            else:
                msg = Fore.RED + msg + Fore.RESET
            overall_status = overall_status and status
            results.append(cls.result_nt(status, msg, exception))

        if not overall_status:
            cls.print_results(results)
            return False
        if verbose:
            cls.print_results(results)
        return True

def add_tags_to_png_file(fpath):
    try:
        info = create_file_info(fpath)
        png_image = PngImageFile(open(fpath, 'rb'))
        png_info = PngInfo()
        for k, v in info.items():
            png_info.add_text(k, v)
        png_image.save(fpath, pnginfo=png_info)
    except (Exception, OSError):
        print("WARNING: Could not add debug info to file '{}'.".format(fpath))
        traceback.print_exc()

def add_tags_to_svg_file(fpath):
    try:
        ElementTree.register_namespace('xlink', "http://www.w3.org/1999/xlink")
        et = ElementTree.parse(fpath)
        root_ns = re.compile('\{([^}]+)\}.*').findall(et.getroot().tag)
        ElementTree.register_namespace('', root_ns[0])
        info = create_file_info(fpath)
        for k, v in info.items():
            new_tag = ElementTree.SubElement(et.getroot(), 'text')
            new_tag.text = "{}: {}".format(k, v)
            new_tag.attrib['style'] = 'font-size:0'
        et.write(fpath)
    except Exception:
        print("WARNING: Could not add debug info to file '{}'.".format(fpath))
        traceback.print_exc()

def create_file_info(fpath):
    info = {}
    info['file_path'] = fpath
    info['git_commit_id'] = subprocess.check_output('git rev-parse HEAD'.split(' ')).strip().decode('utf-8')
    working_dir_state = subprocess.check_output('git status --porcelain'.split(' ')).strip().split(b'\n')
    info['git_staged_count'] = str(sum([1 if x.startswith(b'M') else 0 for x in working_dir_state]))
    info['git_not_staged_count'] = str(sum([1 if x.startswith(b' M') else 0 for x in working_dir_state]))
    info['git_untracked_count'] = str(sum([1 if x.startswith(b'??') else 0 for x in working_dir_state]))
    return info


def savefig(f: matplotlib.figure.Figure, fpath: str, tight: bool=True, details: str=None, **kwargs):
    if tight:
        if details:
            add_parameter_details(f, details, -0.4)
        f.savefig(fpath, bbox_inches='tight', **kwargs)
    else:
        if details:
            f.subplots_adjust(bottom=0.2)
            add_parameter_details(f, details, 0.1)
        f.savefig(fpath, **kwargs)
    if fpath.endswith('png'):
        add_tags_to_png_file(fpath)
    if fpath.endswith('svg'):
        add_tags_to_svg_file(fpath)


def add_parameter_details(f: matplotlib.figure.Figure, details: str, y: float):
    if not details:
        return
    string = ""
    elements = ["{} = {}".format(k, v) for k, v in json.loads(details).items()]
    acc = 0
    for el in elements[:-1]:
        string = string + el + ','
        acc += len(el) + 2
        if acc > 60:
            string = string + '\n'
            acc = 0
        else:
            string = string + ' '
    string += elements[-1]
    f.text(0.05, y, string)


def load_holoviews(fpath):
    from holoviews.core.io import Unpickler
    return Unpickler.load(fpath)

def save_holoviews(fpath, obj, save_components=False, save_doc=False, save_html=True, save_pickle=False, save_item=False):
    from holoviews.core.io import Pickler
    import holoviews as hv
    obj.opts(title="")

    if fpath.endswith('.html'):
        fpath = ".".join(fpath.split(".")[:-1])

    if save_pickle:
        print("saving {}.hvz".format(fpath))
        with open(fpath + '.hvz', 'wb') as f:
            Pickler.save(obj, f)

    if save_item:
        print("saving {}-item.json".format(fpath))
        from bokeh.embed import json_item
        p = hv.render(obj, backend='bokeh')
        item_json = json_item(p)
        with open(fpath + '-item.json', 'w') as f:
            json.dump(item_json, f, indent=2)

    if save_doc:
        print("saving {}-doc.json".format(fpath))
        from bokeh.document import Document
        p = hv.render(obj, backend='bokeh')
        doc = Document()
        doc.add_root(p)
        doc_json = doc.to_json()
        with open(fpath + '-doc.json', 'w') as f:
            json.dump(doc_json, f, indent=2)

    if save_components:
        print("saving {}.{{script|div}}".format(fpath))
        from bokeh.embed import components
        p = hv.render(obj, backend='bokeh')
        script, div = components(p)
        with open(fpath + '.script', 'w') as f:
            f.write(script)
        with open(fpath + '.div', 'w') as f:
            f.write(div)

    if save_html:
        print("saving {}.html".format(fpath))
        hv.save(obj, fpath + ".html")

def pjoin(listo):
    return os.path.sep.join(listo)

def create_desc(df):
    return OrderedDict([
        ('time', str(dt.now())),
        ('columns', df.columns.tolist()),
        ('types', df.dtypes.apply(lambda x: x.name).tolist()),
        ('index-name', str(df.index.name)),
        ('index-type', df.index.dtype.name),
        ('ncols', df.shape[1]),
        ('nrows', df.shape[0]),
        ('ascending_index', df.index.is_monotonic_increasing),
        ('descending_index', df.index.is_monotonic_decreasing)
    ])

def write_desc(dicto, fpath):
    try:
        with open(fpath, 'w') as f:
            json.dump(dicto, f)
    except:
        eprint("Could not write description file.")

def load_desc(fpath):
    fail = False
    if os.path.exists(fpath) and os.path.isfile(fpath):
        try:
            with open(fpath, 'r') as f:
                return json.load(f), fail
        except:
            fail = True
    return None, fail

def write_desc(dicto, fpath):
    try:
        with open(fpath, 'w') as f:
            json.dump(dicto, f, indent=2)
    except:
        eprint("Could not write description file.")

def equal_desc(dict_old, dict_new):
    if len(set(dict_old.keys()).symmetric_difference(set(dict_new.keys()))) > 0:
        print("Desc: change in description format")
        return False
    if not type(dict_old['columns']) is list:
        print("Desc: columns1")
        return False
    if len(set(dict_old['columns']).symmetric_difference(set(dict_new['columns']))) > 0:
        print("Desc: columns2")
        return False
    if not type(dict_old['types']) is list or len(dict_old['types']) != len(dict_old['types']):
        print("Desc: types1")
        return False
    if not all([a == b for a, b in zip(dict_old['types'], dict_new['types'])]):
        print("Desc: types2")
        return False
    if not type(dict_old['ncols']) is int or dict_old['ncols'] != dict_new['ncols']:
        print("Desc: ncols")
        return False
    if not type(dict_old['nrows']) is int or dict_old['nrows'] != dict_new['nrows']:
        print("Desc: nrows")
        return False
    if not type(dict_old['index-name']) is str or dict_old['index-name'] != dict_new['index-name']:
        print("Desc: index-name")
        return False
    if not type(dict_old['index-type']) is str or dict_old['index-type'] != dict_new['index-type']:
        print("Desc: index-type")
        return False
    if not type(dict_old['ascending_index']) is bool or \
            dict_old['ascending_index'] != dict_new['ascending_index']:
        print("Desc: asc. index")
        return False
    if not type(dict_old['descending_index']) is bool or \
            dict_old['descending_index'] != dict_new['descending_index']:
        print("Desc: desc. index")
        return False
    return True


def convert_to_time(string):
    return pd.to_datetime(string, format="%Y-%m-%d %H:%M:%S.%f")


def read_df(fpath: str, silent: bool=False, **kwargs):
    if not silent:
        print("Loading dataframe '{}'".format(fpath))
    ext = os.path.splitext(fpath)[-1]
    if ext == ".h5":
        return pd.read_hdf(fpath)
    if ext == ".csv":
        return pd.read_csv(fpath, **kwargs)
    if ext == ".pickle":
        return pd.read_pickle(fpath, **kwargs)
    raise Exception("No reader for: " + ext)

def hdf_get_metadata(fpath):
    store = pd.HDFStore(fpath)
    try:
        metadata = store.get_storer(HDF_NAMESPACE).attrs.metadata
    except KeyError:
        raise
    finally:
        store.close()
    return metadata

def _write_df(df: pd.DataFrame, fpath: str, **kwargs):
    print("Writing dataframe '{}'".format(fpath))
    ext = os.path.splitext(fpath)[-1]
    if ext == ".h5":
        info = create_file_info(fpath)
        store = pd.HDFStore(fpath)
        store.put(HDF_NAMESPACE, df, format='table')
        store.get_storer(HDF_NAMESPACE).attrs.metadata = info
        store.close()
        return
    if ext == ".csv":
        df.to_csv(fpath, **kwargs)
        return
    if ext == ".pickle":
        df.to_pickle(fpath, **kwargs)
        return
    raise Exception("No writer for: " + ext)

def write_df(df, fpath, constraints=None, desc=False, **kwargs):
    if constraints:
        print("Checking constraints for '{}'".format(fpath))
        if not ConstraintChecker.check_constraints(constraints):
            ext = os.path.splitext(fpath)[-1]
            tmp_path = tempfile.mktemp(ext)
            _write_df(df, tmp_path)
            raise Exception("ERROR: Some constraints failed.")

    if desc:
        json_path = os.path.splitext(fpath)[0] + '.json'
        desc_new = create_desc(df)
        desc_old, fail = load_desc(json_path)
        if fail:
            print("Could not read '{}'".format(json_path))
        if desc_old and not equal_desc(desc_new, desc_old):
            ext = os.path.splitext(fpath)[-1]
            tmp_path = tempfile.mktemp(ext)
            _write_df(df, tmp_path)
            raise Exception("ERROR: The format of the data frame changed. If intended please delete JSON file.")

        print("Writing description of dataframe '{}'".format(json_path))
        write_desc(desc_new, json_path)

    _write_df(df, fpath)


def pdf(data, col, count_col=None):
    data = data.copy()
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
        linestyle="-", marker="", set_ylim=True, **kwargs):
    f = None
    if not ax:
        f, ax = plt.subplots()
    series.plot(ax=ax, linestyle=linestyle, marker=marker, **kwargs)
    if ylog or loglog:
        ax.set_yscale('symlog')
    if xlog or loglog:
        ax.set_xscale('symlog')

    if not ylog and not loglog and set_ylim:
        start = 0 #int(series.iloc[0] * 10)
        ticks = [x/10.0 for x in range(start, 11, 2)]
        ax.set_yticks(ticks)
        ax.set_yticklabels(["{} %".format(int(x*100)) for x in ticks])

    #ax.set_xlabel(col.lower().replace('_', ' '))
    ax.set_ylabel("percentage")
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
        carryover=False, draw_bars=True, cmap='CMRmap', **kwargs):
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
    bar_col = "silver"


    # create a 2 x 3 subplot field
    # TODO: disable option
    if draw_bars:
        fig = plt.figure(frameon=False)
        gs = gridspec.GridSpec(2, 3, width_ratios=[5, 1, 1], height_ratios = [1, 5], figure=fig)
        ax_center = plt.subplot(gs[1,0], frameon=False)
        ax_top = plt.subplot(gs[0,0], frameon=False, sharex=ax_center)
        ax_right = plt.subplot(gs[1,1], frameon=False)
        ax_right2 = plt.subplot(gs[1,2], frameon=False, sharex=ax_center)
    else:
        fig = plt.figure()
        gs = gridspec.GridSpec(2, 1, height_ratios=[9, 1], width_ratios = [1], figure=fig)
        ax_center = plt.subplot(gs[0,0], frameon=False)
        ax_right2 = plt.subplot(gs[1,0], frameon=False)

    # prepare data
    matrix = data.pivot(key1, key2, values=values)
    if carryover:
        matrix = matrix.fillna(method='ffill')
    else:
        matrix = matrix.fillna(0)

    if draw_bars:
        top = data.groupby(key1)[values].sum()
        right = data.groupby(key2)[values].sum()[::-1]

    # plot
    hm_data = matrix.T
    if log:
        norm = SymLogNorm(vmin=hm_data.min().min(), vmax=hm_data.max().max(),
                linthresh=1)
        hm = sns.heatmap(hm_data, ax=ax_center, cbar=False, cmap=cmap,
                norm=norm, linewidths=0, **kwargs)
    else:
        hm = sns.heatmap(hm_data, ax=ax_center, cbar=False, cmap=cmap, linewidths=0, **kwargs)
    # previously used imshow, didn't work as nicely
    ## ax_center.imshow(matrix.T, aspect='auto')
    if draw_bars:
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
    else:
        plt.colorbar(ax_center.get_children()[0], ax=ax_right2, orientation='horizontal', fraction=1, aspect=40)
        fig.subplots_adjust(hspace=0)
        # naming
        ax_center.set_xlabel(xlabel)
        ax_center.set_ylabel(ylabel)

        # remove axes and ticks where possible
        disable_axes(ax_right2)
        disable_ticks(ax_center)

        fig.tight_layout(pad = 0)
        #fig.patch.set_visible(False)
        return fig, [ax_center], None


def plot_heatmap(*args, **kwargs):
    return heatmap(*args, **kwargs)
