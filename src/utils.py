import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


# calculate gaussian kde estimate for a dataset
def kde(mu, tau, bbox=[-5, 5, -5, 5], save_file="", xlabel="", ylabel="", cmap='Blues'):
    values = np.vstack([mu, tau])
    # print('values shape {}'.format(values.shape))
    kernel = stats.gaussian_kde(values)

    fig, ax = plt.subplots()
    ax.axis(bbox) # set axis range by [xmin, xmax, ymin, ymax]
    ax.set_aspect(abs(bbox[1]-bbox[0])/abs(bbox[3]-bbox[2])) # set axis value ratio manually to get equal length
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    xx, yy = np.mgrid[bbox[0]:bbox[1]:300j, bbox[2]:bbox[3]:300j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    # print('positions shape: {}'.format(positions.shape))
    f = np.reshape(kernel(positions).T, xx.shape)
    cfset = ax.contourf(xx, yy, f, cmap=cmap)

    if save_file != "":
        plt.savefig(save_file, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
