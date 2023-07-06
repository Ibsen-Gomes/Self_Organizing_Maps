
# versao 0.2.a
# Imports


import numpy as np

# --------------------------------------------------------------------------- #

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import transforms

# --------------------------------------------------------------------------- #

import pandas as pd

import las2

# --------------------------------------------------------------------------- #

import os # importar_pasta

# --------------------------------------------------------------------------- #

class plotagem:
    
    def __init__(self, n, padrao_usuario = False):
        
        # =============================== #
        
        self.padrao_usuario = {
            #_____________________#
            # imagem
            
            'eixoy':True,
            'comprimento':6,
            'altura':5,
            'resolucao':70,
            'titulo_geral':'',
            'titulo_geral_fonte':16,
            'fundo_cor':'w',
            'fundo_transparencia':0.5,
            'plot_fundo_cor':'w',
            'plot_fundo_transparencia':1.0,
            'padrao_escuro':False,
            #_____________________#
            # curvas
            
            'linha_estilo' : '-',
            'linha_espessura':1,
            'linha_nome':' ',
            'titulo' : '',
            'titulo_fonte':13,
            'cor':'grey',
            'cor_lista':False, # plot m
            'descricao_x':'-',
            'descricao_y':'-',
            'descricao_fonte':13,
            'legenda_simples':False,
            'curva_limite':False,
            'minimo_x':False,
            'maximo_x':False,
            'minimo_y':False,
            'maximo_y':False,
            'maximos_x':False,
            'maximos_y':False
        }
        
        if padrao_usuario:
            
            for i in padrao_usuario:
                
                self.padrao_usuario[i] = padrao_usuario[i]
        
        # =============================== #
        
        self.ax = [0]*n # número de tracks
        self.lab = []   # número de elementos de legenda
        
        self.fig, (self.ax) = plt.subplots(1,n,sharey=self.padrao_usuario['eixoy'],figsize=(
            self.padrao_usuario['comprimento'],
            self.padrao_usuario['altura']),
            dpi=self.padrao_usuario['resolucao'])
        
        self.fig.suptitle(self.padrao_usuario['titulo_geral'], fontsize=self.padrao_usuario['titulo_geral_fonte'])
        
        self.fig.patch.set_facecolor(self.padrao_usuario['fundo_cor'])
        self.fig.patch.set_alpha(self.padrao_usuario['fundo_transparencia'])
        
        self.cor_plot_fundo = self.padrao_usuario['plot_fundo_cor']
        self.transparencia_plot_fundo = self.padrao_usuario['plot_fundo_transparencia']
        
    # ------------------------------------------------------------------------- #
    # definindo definicoes de usuarios locais
    
    def padrao_local(self,padrao_entrada,padrao_base = False):
        
        if padrao_base:
            padrao_base = padrao_base.copy()
        else:
            padrao_base = self.padrao_usuario.copy()
        
        if padrao_entrada:
        
            padrao_local = padrao_base.copy()

            for i in padrao_entrada:

                padrao_local[i] = padrao_entrada[i]
        
        else:
            
            padrao_local = padrao_base.copy()
            
        return padrao_local
        
    # ------------------------------------------------------------------------- #
    
    def max_min_locais(self,X,Y,padrao_entrada):
        
        if padrao_entrada['maximo_x']:
            maxx = padrao_entrada['maximo_x'];
        else:
            maxx = np.nanmax(X)
            
        # =============================== #
            
        if padrao_entrada['minimo_x']:
            minx = padrao_entrada['minimo_x'];
        else:
            minx = np.nanmin(X)
            
        # =============================== #
        
        if padrao_entrada['maximo_y']:
            maxy = padrao_entrada['maximo_y'];
        else:
            maxy = np.nanmax(Y)
            
        # =============================== #
            
        if padrao_entrada['minimo_y']:
            miny = padrao_entrada['minimo_y'];
        else:
            miny = np.nanmin(Y)
            
        # =============================== #
        
        return [minx,maxx,miny,maxy]
        
            
    # ------------------------------------------------------------------------- #
    
    def plot_s(self,
               indice,
               X,
               Y,
               padrao_local = False
            ):

            
        padrao = self.padrao_local(padrao_local).copy()
        
        """plot simples"""
        
        min_max_values = self.max_min_locais(X,Y,padrao)
        minx = min_max_values[0]
        maxx = min_max_values[1]
        miny = min_max_values[2]
        maxy = min_max_values[3]
        
        lab_s = self.ax[indice].plot(X,Y,c = padrao['cor'],
                                     ls = padrao['linha_estilo'],
                                     label = padrao['linha_nome'],
                                     linewidth = padrao['linha_espessura'])
        
        self.ax[indice].grid()
        self.ax[indice].set_ylim(maxy,miny)
        self.ax[indice].set_xlim(minx,maxx)
        self.ax[indice].set_title(padrao['titulo'], fontsize=padrao['titulo_fonte'])
        if indice == 0:
            self.ax[indice].set_ylabel(padrao['descricao_y'], fontsize=padrao['descricao_fonte'])
        self.ax[indice].set_xlabel(padrao['descricao_x'], fontsize=padrao['descricao_fonte'])
        
        self.ax[indice].patch.set_facecolor(self.padrao_usuario['plot_fundo_cor'])
        self.ax[indice].patch.set_alpha(self.padrao_usuario['plot_fundo_transparencia'])
        
        if padrao['legenda_simples']:
            self.ax[indice].legend(loc = padrao['legenda_simples'])
        
        self.lab = self.lab + lab_s
        
    # ------------------------------------------------------------------------- #
        
    def plot_m(self,
               indice,
               XX,
               Y,
               padrao_local = False
        ):
        
        padrao = self.padrao_local(padrao_local).copy()
        
        """plot multiplo"""
        
        min_max_values = self.max_min_locais(Y,Y,padrao)
        miny = min_max_values[2]
        maxy = min_max_values[3]
        
        if padrao['cor_lista']:
            crs = padrao['cor_lista'].copy()
        else:
            crs = ['b']*len(XX)
            
        self.ax[indice].grid()
        self.ax[indice].set_ylim(maxy,miny)

        self.ax[indice].set_title(padrao['titulo'], fontsize=padrao['titulo_fonte'])
        
        for i in range(len(XX)):
            lab_m = self.ax[indice].plot(XX[i],Y,c = crs[i],
                                     ls = padrao['linha_estilo'],
                                     label = padrao['linha_nome'],
                                     linewidth = padrao['linha_espessura'])
            
            self.lab = self.lab + lab_m
        
        if indice == 0:
            self.ax[indice].set_ylabel(padrao['descricao_y'], fontsize=padrao['descricao_fonte'])
        self.ax[indice].set_xlabel(padrao['descricao_x'], fontsize=padrao['descricao_fonte'])
        
        self.ax[indice].patch.set_facecolor(self.padrao_usuario['plot_fundo_cor'])
        self.ax[indice].patch.set_alpha(self.padrao_usuario['plot_fundo_transparencia'])
        
        if padrao['legenda_simples']:
            self.ax[indice].legend(loc = padrao['legenda_simples'])
        
    # ------------------------------------------------------------------------- #
        
    def plog_s(self,indice,X,Y,
               padrao_local = False
            ):
        
        padrao = self.padrao_local(padrao_local).copy()
        
        """plot simples log"""
        
        min_max_values = self.max_min_locais(X,Y,padrao)
        minx = min_max_values[0]
        maxx = min_max_values[1]
        miny = min_max_values[2]
        maxy = min_max_values[3]
        
        lab_log_s = self.ax[indice].semilogx(X,Y,c = padrao['cor'],
                                     ls = padrao['linha_estilo'],
                                     label = padrao['linha_nome'],
                                     linewidth = padrao['linha_espessura'])
        
        self.ax[indice].grid()
        self.ax[indice].set_ylim(maxy,miny)
        self.ax[indice].set_xlim(minx,maxx)
        self.ax[indice].set_title(padrao['titulo'], fontsize=padrao['titulo_fonte'])

        if indice == 0:
            self.ax[indice].set_ylabel(padrao['descricao_y'], fontsize=padrao['descricao_fonte'])
        self.ax[indice].set_xlabel(padrao['descricao_x'], fontsize=padrao['descricao_fonte'])
        
        self.ax[indice].patch.set_facecolor(self.padrao_usuario['plot_fundo_cor'])
        self.ax[indice].patch.set_alpha(self.padrao_usuario['plot_fundo_transparencia'])
        
        if padrao['legenda_simples']:
            self.ax[indice].legend(loc = padrao['legenda_simples'])
        
        self.lab = self.lab + lab_log_s
        
    # ------------------------------------------------------------------------- #
        
    def plot_l(self,
               indice,
               litologia,
               Y,
               relacao_cor,
               padrao_local = False
              ):
        
        padrao = self.padrao_local(padrao_local).copy()
        
        """plot litologia"""
        
        variavel_booleana = False
        
        try:
            variavel_booleana = bool(padrao['curva_limite'])
        except:
            variavel_booleana = bool(padrao['curva_limite'].any())

        if variavel_booleana:
            
            curva_limite = padrao['curva_limite'].copy()
            min_max_values = self.max_min_locais(curva_limite,Y,padrao)
            minx = min_max_values[0]
            maxx = min_max_values[1]
            miny = min_max_values[2]
            maxy = min_max_values[3]
        
        else:
            
            curva_limite = [100.0]*len(Y)
            min_max_values = self.max_min_locais(curva_limite,Y,padrao)
            minx = 0.0
            maxx = 90.0
            miny = min_max_values[2]
            maxy = min_max_values[3]

        codigos = []
        for i in relacao_cor:
            codigos.append(i)
            
        # =============================== #
        
        num_cores = len(codigos)
        
        curva_de_base = [minx]*len(curva_limite)
        
        matriz_litologias = np.array([curva_de_base]*num_cores)
        
        for j in range(num_cores):
            for i in range(len(matriz_litologias[j])):
                if litologia[i] == codigos[j] and  ~np.isnan(curva_limite[i]):
                    matriz_litologias[j][i] = curva_limite[i]
                    
        # =============================== #
        
        for i in relacao_cor:
            self.lab = self.lab + [mpatches.Patch(label=relacao_cor[i][1],color=relacao_cor[i][0])]
        
        # =============================== #
        
        if padrao['linha_espessura'] == self.padrao_usuario['linha_espessura']:
            padrao['linha_espessura'] = 0.00
        
        for i in range(num_cores):
            self.ax[indice].plot(matriz_litologias[i],Y,c = relacao_cor[codigos[i]][0],linewidth = padrao['linha_espessura'])
            self.ax[indice].fill_betweenx(Y,curva_de_base, matriz_litologias[i], facecolor=relacao_cor[codigos[i]][0],
                                          label=relacao_cor[codigos[i]][1])
            
        self.ax[indice].set_ylim(maxy,miny)
        self.ax[indice].set_xlim(minx,maxx)
        if indice == 0:
            self.ax[indice].set_ylabel(padrao['descricao_y'], fontsize=padrao['descricao_fonte'])
        self.ax[indice].set_xlabel(padrao['descricao_x'], fontsize=padrao['descricao_fonte'])
        self.ax[indice].set_title(padrao['titulo'], fontsize=padrao['titulo_fonte'])
        self.ax[indice].set_xticks([])
        
        self.ax[indice].patch.set_facecolor(self.padrao_usuario['plot_fundo_cor'])
        self.ax[indice].patch.set_alpha(self.padrao_usuario['plot_fundo_transparencia'])
            
    # ------------------------------------------------------------------------- #
    
    def plot_l2(self,
           indice,
           litologia,
           Y,
           relacao_cor,
           padrao_local = False
          ):
        
        # =============================== #
        
        cores = {}
        ul = []
        for i in relacao_cor:
            cores[i] = relacao_cor[i][0]
            self.lab = self.lab + [mpatches.Patch(label=relacao_cor[i][1],color=relacao_cor[i][0])]
            ul.append(i)
            
        # =============================== #
        
        padrao = self.padrao_local(padrao_local).copy()
        
        """plot litologia em matriz"""
        
        min_max_values = self.max_min_locais(Y,Y,padrao) # !!! esta redundante !!!
        miny = min_max_values[2]
        maxy = min_max_values[3]    
        
        # =============================== #
        ul = np.array(ul) #np.unique(litologia)
        prof_ = (np.repeat(Y, 3)[2:] + np.repeat(Y, 3)[:-2])/2.0
        lito_ = np.repeat(litologia, 3)[1:-1]
        transformacao = transforms.blended_transform_factory(self.ax[indice].transAxes,self.ax[indice].transData)        
        for l in ul:
            obj = self.ax[indice].fill_betweenx(prof_, 0, 1,where = lito_==l, color = cores[l],transform = transformacao)
            
        # =============================== #
            
        self.ax[indice].set_ylim(maxy,miny)
        if indice == 0:
            self.ax[indice].set_ylabel(padrao['descricao_y'], fontsize=padrao['descricao_fonte'])
        self.ax[indice].set_xlabel(padrao['descricao_x'], fontsize=padrao['descricao_fonte'])
        self.ax[indice].set_title(padrao['titulo'], fontsize=padrao['titulo_fonte'])
        self.ax[indice].set_xticks([])
        
        self.ax[indice].patch.set_facecolor(self.padrao_usuario['plot_fundo_cor'])
        self.ax[indice].patch.set_alpha(self.padrao_usuario['plot_fundo_transparencia'])
    
    # ------------------------------------------------------------------------- #
    
    def legenda(self,
                padrao_local = False
               ):
        
        self.padrao_legenda = {
            
            'ancoragem':(0., 1.10, 2.32, .102),
            'ordem':False,
            'fonte':13,
            'transparencia':0.9,
            'colunas':4,
            'modo':'expand',
            'borda':0.0,
            'indice':0
        }
        
        padrao = self.padrao_local(padrao_local,self.padrao_legenda).copy()
        
        if padrao['ordem']:
            novo_lab = []
            for i in padrao['ordem']:
                novo_lab.append(self.lab[i])
        else:
            novo_lab = self.lab
        
        self.ax[padrao['indice']].legend(handles=novo_lab, bbox_to_anchor=padrao['ancoragem'],
                               loc=0,
                               ncol=padrao['colunas'],
                               mode=padrao['modo'],
                               borderaxespad=padrao['borda'],
                               fontsize=padrao['fonte']).get_frame().set_alpha(padrao['transparencia'])
    
    # ------------------------------------------------------------------------- #
    
    def mostrar(self):
        plt.show()
        
    # ------------------------------------------------------------------------- #
        
    def salvar(self,caminho,transparencia = True,resolucao = 72):
        self.fig.savefig(caminho, transparent=transparencia, dpi = resolucao, bbox_inches="tight")
        
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
    
    def poco_info(arquivo,apelidos):
    
        poco = las2.read(arquivo)

        coordenadas = {}

        for k in apelidos:
            for j in apelidos[k]:
                for i in poco['well']:
                    if i['mnemonic'] == j:
                        coordenadas[k] = i['value']

        return coordenadas
    
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
    
    def pizza(proporcoes,
              info,
              posicao = '%5.0f%%',
              fonte = 18,
              cor = 'k',
              tamanho = (6,6),
              sombra = False,
              angulo = 90,
              salvar = False
             ):

        novas_proporcoes = proporcoes.copy()

        valores = []
        cores = []
        nomes = []

        for i in proporcoes:
            valores.append(proporcoes[i])
            cores.append(info[i][0])
            nomes.append(info[i][1])

        fig1, ax1 = plt.subplots()
        fig1.set_size_inches(tamanho)

        ax1.pie(valores,
                shadow=sombra,
                startangle=angulo,
                labels=nomes,
                autopct=posicao,
                colors=cores,
                textprops={'fontsize': fonte,'color':cor})

        ax1.axis('equal')
        if salvar:
            plt.savefig(salvar[0], transparent=True, dpi = salvar[1], bbox_inches="tight")
        plt.show()
    
    # ============================================ #
    
    def confusao(lit_1,lit_2,label_1 = False,label_2 = False,log=False,tipo="numerico",salvar = False):

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
                        MF[j,i] = round((M1[j,i])/(tamanho),3)

            # ________________________ #

            if tipo == "proporcao_linha": # proporcao em funcao da linha
                M1 = np.array(M1,float)
                MF = M1.copy()

                for j in range(linhas):
                    soma = sum(M1[j])
                    for i in range(colunas):
                        MF[j,i] = round((M1[j,i])/(soma),3)

            # ________________________ #

            if tipo == "proporcao_coluna": # proporcao em funcao da coluna
                M1 = np.array(M1,float)
                MF = M1.copy()

                for i in range(colunas):
                    soma = sum(MF[:,i])
                    for j in range(linhas):
                        MF[j,i] = round((M1[j,i])/(soma),3)

            # ::::::::::::::::::::::::::::::::::::::::::::::: #
            # Tabela e gráficos

            fig = plt.figure(figsize=(6,1))

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

            if salvar:
                plt.savefig(salvar, bbox_inches="tight" )
            else:
                plt.show()

    # ============================================ #
        
    def analise_dispersao(logs,logs_info,lito_log,lito_log_info, titulo = '', titulo_fonte = 16,
                          posicao = False,salvar = False,multi_histogram = False,legenda = False
                         ):
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

        if legenda:

            std_legenda = {
                'posicao':(-1.7,1.0,2.0,-3.5),
                'fonte':16,
                'transparencia':0.9,
                'colunas':4,
                'modo':"expand",
                'borda':0.0
            }

            for i in legenda:

                std_legenda[i] = legenda[i]

            lab = []
            for i in lito_log_info:
                    lab = lab + [mpatches.Patch(label=lito_log_info[i][1],color=lito_log_info[i][0])]

        # =============================== #

        fig = plt.figure(figsize=(16,16))
        fig.suptitle(titulo, fontsize=titulo_fonte, y=0.91)
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
                    if posicao:
                        ax.annotate('('+alfabeto[l-1]+')', xy=(posicao[0], posicao[1]), xycoords='axes fraction',
                                    fontsize = posicao[4],horizontalalignment='right', verticalalignment='bottom')

                if k1 != k2:
                    ax = fig.add_subplot(MX,MX, l)
                    D1 = local_data[k1]
                    D2 = local_data[k2]
                    if posicao:
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
                    plt.title(local_data_names[k2],fontsize = titulo_fonte)

                if l % (MX)-1 == 0:
                    plt.ylabel(local_data_names[k1],fontsize = titulo_fonte)

            if legenda:
                if k1 == 0:
                    ax.legend(handles=lab, bbox_to_anchor=std_legenda['posicao'],
                              loc=0, ncol=std_legenda['colunas'], mode=std_legenda['modo'],
                              borderaxespad=std_legenda['borda'],
                              fontsize=std_legenda['fonte']).get_frame().set_alpha(std_legenda['transparencia'])

        if salvar:
            plt.savefig(salvar[0], transparent=True, dpi = salvar[1], bbox_inches="tight")
        else:
            plt.show()

    
    
