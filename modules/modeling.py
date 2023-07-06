
import random
import numpy as np

import pylab as py
import matplotlib.pyplot as plt
import IPython as ipw
from IPython.display import display
from matplotlib import interactive
from matplotlib import style
from matplotlib.widgets import Cursor, Button

def drilling(ma):
    
    # funcao de callback:
    def click(event):
        #---- append list with picked values -----:
        x.append(event.xdata)
        y.append(event.ydata)
        print('cota:',y)
        print('coordenada H.:',x)
        plotx.append(event.xdata)
        ploty.append(event.ydata)
            
        #-------------- plot data -------------------: 
        line.set_color('k')
        line.set_marker('o')
        line.set_linestyle('None')
        line.set_data(plotx, ploty)
        
       # ax.figure.canvas.draw()     
        
    # ----- for the case of new clicks -------:
    x = []
    y = []
    
    plotx = []
    ploty = []
   # ----------------- cleaning line object for plotting ------------------:
    fig, ax =plt.subplots()
    line, = ax.plot([],[])
         
    # --------- cursor use for better visualization ------------- :
    ax.set_title("Geological Section")
    cursor=Cursor(ax,horizOn=True,vertOn=True, color='black',useblit=True,linewidth=3.0)
    #ax.style.use(['classic'])  
    plt.style.use(['classic']) #fundo cinza
    plt.imshow(ma)
    plt.grid()

# ------------ Hack because Python 2 doesn't like nonlocal variables that change value -------:
# Lists it doesn't mind.
    picking = [True]
    fig.canvas.mpl_connect('button_press_event', click )
    plt.show()
     
    return x,y