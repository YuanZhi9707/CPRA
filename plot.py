import numpy as np
from matplotlib import pyplot as plt
import pylab as pl

x1 = [5, 6, 7, 8, 9]  # Make x, y arrays for each graph
y1 = [0.7913, 0.8129, 0.8135, 0.8136, 0.8137]
x2 = [5, 6, 7, 8, 9]
y2 = [0.8513, 0.8729, 0.8735, 0.8736, 0.8737]

pl.plot(x1, y1, 'r', label='100H')  # use pylab to plot x and y : Give your plots names
pl.plot(x2, y2, 'g', label='100L')

pl.title('RG')  # give plot a title
pl.xlabel('Numble of blocks')  # make axis labels
pl.ylabel('PSNR')

pl.xlim(5, 9)  # set axis limits
pl.ylim(0.75, 1)
pl.legend()
pl.show()  # show the plot on the screen

