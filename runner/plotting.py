"""plotting methods for pandas DataFrame
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from pandas import scatter_matrix  # nice

def parallel_coordinates(df, name=None, colormap=None, alpha=0.5, 
                         add_cb=True, cb_axes=[0.05, 0, 0.9, 0.05], 
                         normalize=True, 
                         cb_orientation='horizontal', **kwargs):
    """Call to parallel_coordinates + customization
    
    * df : pandas DataFrame
    * name : variable name along which to sort values
    * normalize : True by default
    * colormap : e.g. viridis, inferno, plasma, magma
        http://matplotlib.org/examples/color/colormaps_reference.html
    * add_cb : add a colorbar
        http://matplotlib.org/examples/api/colorbar_only.html
    * cb_axes : tuned for horizontal cb with 5 variables
    * cb_orientation : horizontal or vertical
    * **kwargs : passed to original pandas.tools.plotting.parallel_coordinates
    """
    df = df.dropna()
    if normalize:
        delta = (df - df.mean()) / df.std()  # normalize
    else:
        delta = df

    if name is None:
        name = df.columns[0]

    # insert class variable as new variable
    cls = df[[name]].rename(columns={name:'cls'})
    full = pd.concat([cls, delta], axis=1).sort_values('cls')

    # http://matplotlib.org/examples/api/colorbar_only.html
    # http://matplotlib.org/examples/color/colormaps_reference.html
    cmap = colormap or mpl.cm.viridis
    axes = pd.tools.plotting.parallel_coordinates(full, 'cls', alpha=alpha, 
                                                  colormap=cmap)
    axes.legend().remove()  # remove the legend
    axes.set_ylabel('normalized')

    # add colorbar axis (since no mappable is easily found...)
    if not add_cb:
        return axes

    fig = plt.gcf()
    ax1 = fig.add_axes(cb_axes)
    norm = mpl.colors.Normalize(vmin=cls.values.min(), vmax=cls.values.max())
    cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,
                                    norm=norm, 
                                    orientation=cb_orientation)
    cb1.set_label(name)
    return axes
