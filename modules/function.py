
import numpy as np
import matplotlib.patches as mpatches

# --------------------------------------------------------------------------- #

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import transforms

# --------------------------------------------------------------------------- #

import pandas as pd

def perfurar(ma):
    # ----- for the case of new clicks -------:
    x = []
    y = []
    
    plotx = []
    ploty = []
   # ----------------- cleaning line object for plotting ------------------:
    fig, ax =plt.subplots()
    
    line, = ax.plot([],[])
         
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
        
        ax.figure.canvas.draw()
              
    
    # --------- cursor use for better visualization ------------- :
    cursor=Cursor(ax,horizOn=True,vertOn=True, color='green',useblit=True,linewidth=2.0)
    #ax.style.use(['classic'])  
    py.rcParams['figure.figsize'] = (20.0, 30.0)#new figure dimension
    ax.set_title("Modelo")
    plt.imshow(ma)
    plt.grid()

# ------------ Hack because Python 2 doesn't like nonlocal variables that change value -------:
# Lists it doesn't mind.
    picking = [True]
    fig.canvas.mpl_connect('button_press_event', click )
    plt.show()
    
    return x,y


def RHOmb = [0.0]*np.size(mb,0):
    for i in range(np.size(mb,0)): # Cria um Laço com índice i que varia de 0 até o tamanho de mb menos 1. 
        if dist[i] == codigo['0'][1]:     #declara que quando o índice i for igual a 0 na posição mn ...
            RHOmb[i] = 2.12 # o elemento do vetor assumirá o valor de 2.4...e assim sucessivamente ... 
        if dist[i] == codigo['1'][1]:
            RHOmb[i] = 2.25
        if dist[i] == codigo['2'][1]:
            RHOmb[i] = 2.43
        if dist[i] == codigo['3'][1]:
            RHOmb[i] = 1.8
        if dist[i] == codigo['4'][1]:
            RHOmb[i] = 1.9
        if dist[i] == codigo['5'][1]:
            RHOmb[i] = 2.4
        if dist[i] == codigo['6'][1]:
            RHOmb[i] = 2.55
        if dist[i] == codigo['7'][1]:
            RHOmb[i] = 2.75
        if dist[i] == codigo['8'][1]:
            RHOmb[i] = 2.7
        if dist[i] == codigo['9'][1]:
            RHOmb[i] = 2.45
        prof.append(i) # Preenche o vetor prof com os valores do índice i. Comando append()
return RHOmb = [0.0]*np.size(mb,0)


def analise_dispersao(logs,logs_info,lito_log,lito_log_info,
                      posicao = False,salvar = False,multi_histogram = False,legenda = False, padrao_usuario = False
                     ):
    
    # =============================== #
    
    padrao_programa = {
        
        # tudo o que pode ser alterado
        'titulo':'',
        'titulo_fonte':16,
        'posicao':False,
        'salvar':False,
        'legenda':False,
        'figura_tamanho':(18,18)
    }
    
    if padrao_usuario:
        
        for i in padrao_usuario:
            
            padrao_programa[i] = padrao_usuario[i]
            
    # =============================== #        
    
    MX = len(logs_info)
    
    local_data_names = []
    local_data = []
    
    for i in logs_info:
        local_data_names.append(logs_info[i][0])
        local_data.append(logs[i])
    
    # =============================== #
    
    alfabeto = 'abcdefghijklmnopqrstuvwxyz'
    
    local_lithos_colors = []
    local_lithos_index = []
    MY = len(lito_log_info)
    
    for i in lito_log_info:
        local_lithos_colors.append(lito_log_info[i][0])
        
    for j in lito_log_info:
        mini_index = []
        for i in range(len(lito_log)):
            if lito_log[i] == j:
                mini_index.append(i)
        local_lithos_index.append(mini_index)
    
    # =============================== #
    # separacao litologias
    
    if multi_histogram:
        all_data = []
        for k in logs_info: # por curva
            mini_data = []
            for j in lito_log_info: # por cor
                curve = []
                for i in range(len(lito_log)): # por profundidade
                    if lito_log[i] == j:
                        curve.append(logs[k][i])
                mini_data.append(curve)
            all_data.append(mini_data)
    
    # =============================== #
    # legendas
    
    if padrao_programa['legenda']:
        
        std_legenda = {
            'posicao':(-3.5,4.0,4.0,-8),
            'fonte':16,
            'transparencia':0.9,
            'colunas':4,
            'modo':"expand",
            'borda':0.0
        }
    
    # =============================== #
        
        for i in padrao_programa['legenda']:
            
            std_legenda[i] = padrao_programa['legenda'][i]
        
        lab = []
        for i in lito_log_info:
                lab = lab + [mpatches.Patch(label=lito_log_info[i][1],color=lito_log_info[i][0])]
    
    # =============================== #
    
    fig = plt.figure(figsize=padrao_programa['figura_tamanho'])
    fig.suptitle(padrao_programa['titulo'], fontsize=padrao_programa['titulo_fonte'], y=0.91)
    #plt.rcParams['axes.facecolor'] = 'k'
    # ----------------------------------------------------------- #

    l = 0
    for k1 in range(MX):
        for k2 in range(MX):
            l = l + 1

            if k1 == k2:
                ax = fig.add_subplot(MX,MX, l)
                if multi_histogram:
                    for i in range(MY):
                        ax.hist(all_data[k1][i],multi_histogram[0],color=local_lithos_colors[i],
                                alpha=multi_histogram[1]) ###
                else:
                    ax.hist(local_data[k1],100) ###
                
                
                ax.patch.set_alpha(0.0)
                if padrao_programa['posicao']:
                    ax.annotate('('+alfabeto[l-1]+')', xy=(posicao[0], posicao[1]), xycoords='axes fraction',
                                fontsize = posicao[4],horizontalalignment='right', verticalalignment='bottom')

            if k1 != k2:
                ax = fig.add_subplot(MX,MX, l)
                D1 = local_data[k1]
                D2 = local_data[k2]
                if padrao_programa['posicao']:
                     ax.annotate('('+alfabeto[l-1]+')', xy=(posicao[2], posicao[3]), xycoords='axes fraction',
                                fontsize = posicao[4],horizontalalignment='right', verticalalignment='bottom')
                
                #print(D1,D2)

                for j in range(len(local_lithos_colors)):
                    #print(k1,k2,j)
                    gr = [];dt = []
                    for i in range(len(local_lithos_index[j])):
                        gr.append(D1[local_lithos_index[j][i]])
                        dt.append(D2[local_lithos_index[j][i]])
                    ax.plot(dt,gr,'.',alpha=1,color=local_lithos_colors[j])
                    ax.patch.set_alpha(0.0)

            if l < MX + 1:
                plt.title(local_data_names[k2],fontsize = padrao_programa['titulo_fonte'])

            if l % (MX)-1 == 0:
                plt.ylabel(local_data_names[k1],fontsize = padrao_programa['titulo_fonte'])
                
        if padrao_programa['legenda']:
            if k1 == 0:
                ax.legend(handles=lab, bbox_to_anchor=std_legenda['posicao'],
                          loc=0, ncol=std_legenda['colunas'], mode=std_legenda['modo'],
                          borderaxespad=std_legenda['borda'],
                          fontsize=std_legenda['fonte']).get_frame().set_alpha(std_legenda['transparencia'])
                
    if salvar:
        plt.savefig(salvar[0], transparent=True, dpi = salvar[1], bbox_inches="tight")
    else:
        plt.show()