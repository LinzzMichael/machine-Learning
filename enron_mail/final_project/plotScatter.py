import sys
reload(sys)
sys.setdefaultencoding('gbk')


import matplotlib.pyplot as plt
import numpy as np


def plotScatter(features, labels, x1, x2, xName, yName):
    colors = ["b", "c", "k", "m", "g"]

    for ii, pp in enumerate(labels):
        plt.scatter(int(features[ii][x1]), int(features[ii][x2]), color = colors[int(labels[ii])])

    plt.xlabel(xName)
    plt.ylabel(yName)
    plt.show()



if __name__ == '__main__':
    features = [[1,2],[2,3],[6,6]]
    labels = [0, 1, 1]
    plotScatter(features, labels, 0, 1)
    