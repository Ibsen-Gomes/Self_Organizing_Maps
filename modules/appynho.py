# Imports

import numpy as np

# --------------------------------------------------------------------------- #

import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #

import pandas as pd

import las2

# --------------------------------------------------------------------------- #

import os # importar_pasta

# --------------------------------------------------------------------------- #

class plotagem:
    
    def __init__(self, n, eixoy=True, comprimento=6, altura=5, dpi=70, titulo = '', titulo_fonte = 16,
                cor_fundo = 'white',transparencia_fundo = 0.5,
                cor_plot_fundo = 'white',transparencia_plot_fundo = 1.0):
        
        self.ax = [0]*n
        self.fig, (self.ax) = plt.subplots(1,n,sharey=eixoy,figsize=(comprimento, altura),
                                 dpi=dpi)
        self.fig.suptitle(titulo, fontsize=titulo_fonte)
        
        self.fig.patch.set_facecolor(cor_fundo)
        self.fig.patch.set_alpha(transparencia_fundo)
        
        self.cor_plot_fundo = cor_plot_fundo
        self.transparencia_plot_fundo = transparencia_plot_fundo
    
    def plot_s(self,indice,X,Y,
             cor='b',estilo_linha = '-',
             descricao_x = 'x',descricao_y = 'y',fonte_descricao = 16,
             titulo = 'titulo',fonte_titulo = 15
            ):
        
        """plot simples"""
        
        self.ax[indice].plot(X,Y,c = cor,ls = estilo_linha)
        self.ax[indice].grid()
        self.ax[indice].set_ylim(max(Y),min(Y))
        self.ax[indice].set_title(titulo, fontsize=fonte_titulo)
        if indice == 0:
            self.ax[indice].set_ylabel(descricao_y, fontsize=fonte_descricao)
        self.ax[indice].set_xlabel(descricao_x, fontsize=fonte_descricao)
        
        self.ax[indice].patch.set_facecolor(self.cor_plot_fundo)
        self.ax[indice].patch.set_alpha(self.transparencia_plot_fundo)
        
    def plot_m(self,indice,XX,Y,cores = False,estilo_linha = '-',
              descricao_x = 'x',descricao_y = 'y',fonte_descricao = 16,
              titulo = 'titulo',fonte_titulo = 15):
        
        """plot multiplo"""
        
        if cores:
            crs = cores.copy()
        else:
            crs = ['b']*len(XX)
        
        for i in range(len(XX)):
            self.ax[indice].plot(XX[i],Y,c = crs[i],ls = estilo_linha)
        
        if indice == 0:
            self.ax[indice].set_ylabel(descricao_y, fontsize=fonte_descricao)
        self.ax[indice].set_xlabel(descricao_x, fontsize=fonte_descricao)
        self.ax[indice].set_title(titulo, fontsize=fonte_titulo)
        self.ax[indice].set_xticklabels([])
        
        self.ax[indice].patch.set_facecolor(self.cor_plot_fundo)
        self.ax[indice].patch.set_alpha(self.transparencia_plot_fundo)
        
    def plog_s(self,indice,X,Y,
             cor='b',estilo_linha = '-',
             descricao_x = 'x',descricao_y = 'y',fonte_descricao = 16,
             titulo = 'titulo',fonte_titulo = 15
            ):
        
        """plot simples"""
        
        self.ax[indice].semilogx(X,Y,c = cor,ls = estilo_linha)
        self.ax[indice].grid()
        self.ax[indice].set_ylim(max(Y),min(Y))
        self.ax[indice].set_title(titulo, fontsize=fonte_titulo)
        if indice == 0:
            self.ax[indice].set_ylabel(descricao_y, fontsize=fonte_descricao)
        self.ax[indice].set_xlabel(descricao_x, fontsize=fonte_descricao)
        
        self.ax[indice].patch.set_facecolor(self.cor_plot_fundo)
        self.ax[indice].patch.set_alpha(self.transparencia_plot_fundo)
        
    def plot_l(self,indice,litologia,Y,relacao_cor,curva_limite,minimo = False,maximo = False,
              descricao_x = '',descricao_y = 'y',fonte_descricao = 16,
              titulo = 'titulo',fonte_titulo = 15, legend=False):
        
        """plot litologia"""

        codigos = []
        for i in relacao_cor:
            codigos.append(i)
            
        if minimo:
            minimo = minimo
        else:
            minimo = min(curva_limite)
            
        if maximo:
            maximo = maximo
        else:
            maximo = max(curva_limite)
        
        num_cores = len(codigos)
        
        matriz_litologias = np.array([[minimo]*len(curva_limite)]*num_cores)
        
        for j in range(num_cores):
            for i in range(len(matriz_litologias[j])):
                if litologia[i] == codigos[j] and  ~np.isnan(curva_limite[i]):
                    matriz_litologias[j][i] = curva_limite[i]
                    
        # =============================== #
        
        for i in range(num_cores):
            self.ax[indice].plot(matriz_litologias[i],Y,c = relacao_cor[codigos[i]][0],lw = 0.1)
            self.ax[indice].fill_betweenx(Y, matriz_litologias[i], facecolor=relacao_cor[codigos[i]][0], label=relacao_cor[codigos[i]][1])
        self.ax[indice].set_ylim(max(Y),min(Y))
        self.ax[indice].set_xlim(minimo,maximo)
        if indice == 0:
            self.ax[indice].set_ylabel(descricao_y, fontsize=fonte_descricao)
        self.ax[indice].set_xlabel(descricao_x, fontsize=fonte_descricao)
        self.ax[indice].set_xticks([])
        self.ax[indice].set_title(titulo, fontsize=fonte_titulo)
        
        self.ax[indice].patch.set_facecolor(self.cor_plot_fundo)
        self.ax[indice].patch.set_alpha(self.transparencia_plot_fundo)
        
        if legend==True:
            self.fig.legend(loc='lower center', fancybox=True, shadow=True, ncol=(3))
    
    def mostrar(self):
        plt.show()
        
    def salvar(self,caminho,transparencia = True):
        self.fig.savefig(caminho, transparent=transparencia)
        
# --------------------------------------------------------------------------- #
# ###
# --------------------------------------------------------------------------- #

class gerenciamento():
    
    def __init__(self):
        
        self.projetos = {}
        
    # ============================================ #
        
    def importar_pasta(caminho_geral,nomes = False,ext = '.las'):

        #------------------------------------------------------------------#
        # vai conter o caminho até os arquivos em geral
        arquivos = []
        # r=root, d=directories, f = files
        for r, d, f in os.walk(caminho_geral):
            for file in f:
                if ext in file:
                    arquivos.append(os.path.join(r, file))

        #------------------------------------------------------------------#
        # arquivos = caminho geral ate os arquivos
        # names = nomes dos poços

        if nomes:

            names = []
            for i in arquivos:
                n1 = i.replace(caminho_geral+'/', '')
                names.append(n1.replace(ext,''))

            return [arquivos,names]

        else:

            return arquivos
        
    # ============================================ #
    
    def importar_las(caminho,apelidos=False):

        campo = {}

        dado_lido = las2.read(caminho)

        nomes = [a['mnemonic'] for a in dado_lido['curve']]
        unidades = [a['unit'] for a in dado_lido['curve']]
        dado = {}
        for i in range(len(nomes)):
            dado[nomes[i]] = dado_lido['data'][i]

        # ------------------------------------ #

        if apelidos:
            dado_final = {}
            for i in dado:
                for j in apelidos:
                    for k in apelidos[j]:

                        if i == k:
                            #print(i,'apelidado de',j)
                            dado_final[j] = dado[i]

            return dado_final

        # ------------------------------------ #

        else:
            return dado
    
    # ============================================ #
    
        
    def importar_csv(caminho,profundidades,mnemonico):

        dado = pd.read_csv(caminho)

        print("cabecalho =",dado.columns.values)

        dado_final = {}

        for i in list(dado.columns.values):
            for j in mnemonico:
                for k in mnemonico[j]:

                    if i == k:
                        print(i,'apelidado de',j)
                        dado_final[j] = list(dado[i])

        lito = dado_final['codigo']
        ptop = dado_final['topo']
        pbot = dado_final['base']

        lito_2 = [0.0]*len(profundidades)

        for j in range(len(ptop)):
            for i in range(len(profundidades)):
                if profundidades[i] >= ptop[j] and profundidades[i] < pbot[j]:
                    lito_2[i] = lito[j]

        return lito_2
    
    # ============================================ #
    
    def importar_dados(caminhos,pocos=False):
            
        # ------------------------------------ #
            
        campo = {}
        for j in range(len(caminhos)):

            dado_lido = las2.read(caminhos[j])

            nomes = [a['mnemonic'] for a in dado_lido['curve']]
            unidades = [a['unit'] for a in dado_lido['curve']]
            dado = {}
            for i in range(len(nomes)):
                dado[nomes[i]] = dado_lido['data'][i]

            campo[caminhos[j]] = [dado,nomes,unidades]
            
        return [nomes,campo]
    
    # ============================================ #
    
    def cropar(profundidade,curvas,topo=0,base=20000,nulos=False):

        novas_curvas = []
        for j in range(len(curvas)):
            curva = []
            profundiade_cropada = []
            for i in range(len(profundidade)):
                if profundidade[i] >= topo and profundidade[i] < base:
                    curva.append(curvas[j][i])
                    profundiade_cropada.append(profundidade[i])
            novas_curvas.append(curva)

        novas_curvas_final = []
        novas_curvas_final.append(profundiade_cropada)
        for i in range(len(curvas)):
            novas_curvas_final.append(novas_curvas[i])

        return novas_curvas_final
    
    # ============================================ #
    
    def cropar_limpo(profundidade,curvas,topo=0,base=20000,nulos=False):

        #nulos_idx = [True]*len(profundidade)

        novas_curvas = []
        for j in range(len(curvas)):
            curva = []
            profundiade_cropada = []
            for i in range(len(profundidade)):
                if profundidade[i] >= topo and profundidade[i] < base:
                    curva.append(curvas[j][i])
                    profundiade_cropada.append(profundidade[i])
            novas_curvas.append(curva)

        novas_curvas_final = []
        novas_curvas_final.append(profundiade_cropada)
        for i in range(len(curvas)):
            novas_curvas_final.append(novas_curvas[i])

        a = np.array(novas_curvas_final).T

        b = a[~np.isnan(a).any(axis=1)]
        if nulos:
            b = b[~np.isin(b,nulos).any(axis=1)]

        return list(b.T)

    # ============================================ #
    
    def cropar_limpo_2(profundidade,curvas,topo=0,base=20000,nulos=False):

        p2 = []
        for j in curvas:
            curva = []
            for i in range(len(curvas[j])):
                if profundidade[i] >= topo and profundidade[i] < base:
                    curva.append (curvas[j][i])

            p2.append(curva)

        a = np.array(p2).T
        b = a[~np.isnan(a).any(axis=1)]
        if nulos:
            b = b[~np.isin(b,nulos).any(axis=1)]

        c = b.T

        log_limpo = {}
        i = 0
        for key in curvas:
            log_limpo[key] = c[i]
            i += 1

        return log_limpo
    
    # ============================================ #
    
# --------------------------------------------------------------------------- #
# ###
# --------------------------------------------------------------------------- #

class visual:
    
    # ============================================ #
    
    def confusao(lit_1,lit_2,label_1 = False,label_2 = False,log=False,tipo="numerico"):

        # ::::::::::::::::::::::::::::::::::::::::::::::: #
        # Definição de variáveis

        s_1 = sorted(list(set(lit_1))) # lista dos elementos de lit_1
        s_2 = sorted(list(set(lit_2))) # lista dos elementos de lit_2

        if log:
            print(s_1)
            print(s_2)

        # ::::::::::::::::::::::::::::::::::::::::::::::: #
        # salvando as labels (loop dos elementos)

        nms_1 = []
        for i in range(len(s_1)):
            if label_1:
                nms_1.append(label_1[int(s_1[i])])
            else:
                nms_1.append(int(s_1[i]))

        # ________________________ #

        nms_2 = []
        for i in range(len(s_2)):
            if label_1:
                if label_2:
                    nms_2.append(label_2[int(s_2[i])])
                else:
                    nms_2.append(label_1[int(s_2[i])])
            else:
                nms_2.append(int(s_2[i]))

        # ::::::::::::::::::::::::::::::::::::::::::::::: #
        # Calculando o erro geral para apresentação

        err = []
        for i in range(len(lit_1)):
            if lit_1[i] == lit_2[i]:
                err.append(1)
            else:
                err.append(0)

        if log:
            print('acerto = ',sum(err),'de',len(err),'equivalente a',(sum(err)/len(err))*100.0,'%')

        # ::::::::::::::::::::::::::::::::::::::::::::::: #
        # calculo dos valores (por dicionário)

        CM = {}
        M1 = []
        for j in range(len(s_1)):
            CM[int(s_1[j])] = {}
            M0 = []
            for i in range(len(s_2)):
                values = []
                for jj in range(len(lit_1)):
                    if lit_1[jj] == int(s_1[j]):
                        if lit_2[jj] == int(s_2[i]):
                            values.append(1)
                        else:
                            values.append(0)

                sv = sum(values)
                CM[int(s_1[j])][s_2[i]] = sv
                M0.append(sv)
            M1.append(M0)

        # ::::::::::::::::::::::::::::::::::::::::::::::: #
        # calculando proporções

        linhas = np.shape(M1)[0]
        colunas = np.shape(M1)[1]
        tamanho = len(lit_1)

        if tipo == "numerico": # numeros de elementos contados (padrão)
            M1 = np.array(M1)
            MF = M1.copy()

        # ________________________ #

        if tipo == "proporcao": # proporcao em funcao do total
            M1 = np.array(M1,float)
            MF = M1.copy()

            for j in range(linhas):
                for i in range(colunas):
                    MF[j,i] = (M1[j,i])/(tamanho)

        # ________________________ #

        if tipo == "proporcao_linha": # proporcao em funcao da linha
            M1 = np.array(M1,float)
            MF = M1.copy()

            for j in range(linhas):
                soma = sum(M1[j])
                for i in range(colunas):
                    MF[j,i] = (M1[j,i])/(soma)

        # ________________________ #

        if tipo == "proporcao_coluna": # proporcao em funcao da coluna
            M1 = np.array(M1,float)
            MF = M1.copy()

            for i in range(colunas):
                soma = sum(MF[:,i])
                for j in range(linhas):
                    MF[j,i] = (M1[j,i])/(soma)

        # ::::::::::::::::::::::::::::::::::::::::::::::: #
        # Tabela e gráficos

        the_table = plt.table(cellText=MF,
                          colWidths=[0.1] * len(lit_2),
                          rowLabels=nms_1,
                          colLabels=nms_2,
                          loc='center')

        the_table.auto_set_font_size(False)
        the_table.set_fontsize(24)
        the_table.scale(4, 4)

        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)

        for pos in ['right','top','bottom','left']:
            plt.gca().spines[pos].set_visible(False)
        plt.show()
        
    # ============================================ #
    
    
