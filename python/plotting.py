import matplotlib
from matplotlib import colors
import numpy as np
import itertools

import matplotlib.pyplot as plt
import matplotlib.patches

def scatter_clusters(ax,locations,clusters,colors):
    keys = list(set(clusters))
    for c in keys:
        indices = np.where(clusters == c)
        #print(c,colors[c],len(indices))
        ax.scatter(locations[indices,0],locations[indices,1],color=colors[c])
        
def create_colors(n):
    #assumes hue [0, 360), saturation [0, 100), lightness [0, 100)
    ret = []
    for j in range(n):
        i = j * 360.0 / n
        hue = i
        saturation = 90 + np.random.random() * 10
        lightness = 50 + np.random.random() * 10

        hue /= 360.0
        saturation /= 100.0
        lightness /= 100.0
        
        color = colors.hsv_to_rgb([hue,saturation,lightness])
        
        ret += [color]
        

    return ret
    
def plot_clustering(ax,locations,nclusters,clusters,cells,probs=None,colors=None):
    n,m = locations.shape

    color_map = create_colors(nclusters)

    #colorsf = get_cmap(nclusters)        
    col = {}

    if colors is None:
        for i in range(nclusters):
            col[i] = color_map[i]
    else:
        for i in range(nclusters):
            if i < len(colors):
                col[i] = colors[i]
            else:
                col[i] = color_map[i]  
            
    if probs is None:
        probs = np.ones(n) *0.6
            
    for i in range(n):
        c = col[clusters[i]]
        #print i,c,probs[i]
        plot_cell(ax,i,locations,cells,probs[i],color=c)
        
        
def fix_cell_limits(cells,xmin,xmax,ymin,ymax):
    n = len(cells)
    
    for i in range(n):
        for v in cells[i]["vertices"]:
            x,y = v
            if x < xmin:
                x = xmin
            elif x > xmax:
                x = xmax
            if y < ymin:
                y = ymin
            elif y > ymax:
                y = ymax
            
            v[0] = x
            v[1] = y
    
def plot_cell(ax,i,locations,cells,alpha,color):
    ax.plot(locations[:,0],locations[:,1],"+")

    volume = cells[i]["volume"]
    #print locations[i],volume

    line_segments = []
    for v in cells[i]["vertices"]:
        x,y = v
        line_segments += [[x,y]]

    line_segments = np.array(line_segments)
    
    #print line_segments
    ax.add_patch(matplotlib.patches.Polygon(line_segments,alpha=alpha,color=color))
    x,y = locations[i,:]
    #ax.annotate(str(i), xy=(x, y))


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,xlabel="X",ylabel="Y",decimals=2):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.round(cm,decimals)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    
if __name__ == "__main__":
    colors = create_colors(5)
    print(colors)
