import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.transforms as tr

## CONFIDENCE ELLIPSE ##
def getConfidenceEllipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if len(x) != len(y):
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = tr.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def computeDensities(events, format, div, combination = 'duo'):
    if combination == 'duo':
        mult = 2
    elif combination == 'trio':
        mult = 3
    elif combination == 'none':
        mult = 1

    time_window = 5000
    X,Y,T = int((128*mult)/div), int(128/div), max(events[:,format.index('t')])
    start = 0
    done = set()
    nb = 0
    temporal_density = [[0 for _ in range(X)] for _ in range(Y)]
    spatial_density  = 0

    for ev in events:
        x = ev[format.index('x')]
        y = ev[format.index('y')]
        t = ev[format.index('t')]
        
        # compute length
        nb += 1

        # compute spatial density
        if t >= (start + time_window):
            nb_timewindows = int( (t - start) // time_window )
            start += time_window * nb_timewindows
            spatial_density += len(done) / ( X*Y )
            done.clear()

        if (x,y) not in done:
            done.add((x,y))
        
        # compute temporal density
        temporal_density[int(y)][int(x)] += 1

    spatial_density  /= int( T // time_window )
    temporal_density = np.sum(list(temporal_density / T)) / (X*Y)

    return spatial_density, temporal_density