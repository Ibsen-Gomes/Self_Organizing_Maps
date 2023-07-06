#-*- coding: utf-8 -*-

# Oração segunda de S. Tomás de Aquino para o estudo:

#Criador Inefável,
#Vós que sois a fonte verdadeira da luz e da ciência,
#derramai sobre as trevas da minha inteligência um raio da vossa claridade.

#Dai-me inteligência para compreender,
#memória para reter,
#facilidade para aprender,
#sutileza para interpretar
#e graça abundante para falar.

#Meu Deus, semeai em mim a semente da vossa bondade.

#Fazei-me pobre sem ser miserável,
#humilde sem fingimento,
#alegre sem superficialidade,
#sincero sem hipocrisia;
#que faça o bem sem presunção,
#que corrija o próximo sem arrogância,
#que admita a sua correção sem soberba;
#que a minha palavra e a minha vida sejam coerentes.

#Concedei-me, Verdade das verdades,
#inteligência para conhecer-Vos,
#diligência para Vos procurar,
#sabedoria para Vos encontrar,
#uma boa conduta para Vos agradar,
#confiança para esperar em Vós,
#constância para fazer a Vossa vontade.

#Orientai, meu Deus, a minha vida;
#concedei-me saber o que me pedis
#e ajudai-me a realizá-lo
#para o meu próprio bem
#e de todos os meus irmãos.

#Amém.

#-------------------------------------------#
#  Organiza as minhas subrotinas dispersas. #
# Carreira,V.R.(2020)                       #
#-------------------------------------------#


#########PACOTES############
import os
import sys
import numpy as np
import pandas as pd
############################



class Debug:
    pass

    def pause():
        '''
        FORTRANIC logical debugging. 
        Just for fortranic beings.
        '''
        programPause = input("Press the <ENTER> key to continue...")
        return

    def stop():
        '''
        FORTRANIC logical debugging. 
        Just for fortranic beings.
        '''
        sys.exit('Stop here!')
        return



class Preprocessing:
    pass

    def Winput():
        '''
        Generical method to read well log *.txt, *.las and *.csv data into a correct pandas Data Frame. 
        Input:
              - File name
              - Number of header lines (optional: use 0 instead)
              - Number of footer lines (optional: use 0 instead)
        Output:
              - df, Well Data into a Pandas Data Frame
              - Data Frame information
        '''
        
        file=str(input("File name="))
        sr=int(input("Header's line numbers="))
        sp=int(input("Footer's line numbers="))
        
        df = pd.read_csv(file , sep='\s+', skiprows=sr ,skipfooter=sp, index_col=0)
        
        #Inverte as linhas do dataframe e reseta os índices:
        df=df[::-1].reset_index()

        return df, print(df.info())
#-----------------------------------------
    def channels(df):# Verificar!!!
        '''
        Return the names of the log drilling channels inside data.
        Input: 
             -  df, Pandas Data Frame
        Output:
             - columns names
        '''

        for col_name in df.columns: 
            print(col_name)
        return df
#------------------------------------------
    def lcounter(channel):
        '''
        Count the number of samples in a rock class:
        Inputs:
           - channel, Pandas Data Frame of codes;
        Outputs:
           - k, counter
           - code, rock class 
        '''
        k=0
        code=int(input('Input code =' ))
        drill=np.asarray(channel)
        for i in range(len(drill)):
            if drill[i] == code:
                k=k+1
        return print('There is',k,'numbers of rock with code',code)
    
#-------------------------------------------
    def noncollapsed(dim,delta,cali,df):
        '''
        Search and filter noncollapsed well parts based on caliper analysis.
        Inputs:
               - dim, well's diameter
               - delta, acceptable well's diameter variation
               - cali, pd.DataFrame that contains caliper channel info
               - df, total dataframe info
        Outputs:
               - filtered data
        OBS: consider to use pd.read_csv method for input channels. 
        '''
        ls = dim + delta
        li = dim - delta
        df=df[( cali >= li) & ( cali <= ls)]

        return df

#--------------------------------------------
    def spurious(df,channel,a,b):
        '''
        Search and filter tool errors. Fixed inspired real experience.
        Inputs:
              - df, Pandas Data Frame
              - channel, channel to be filtered
              - a, could be a real or a dummy value
              - b, coudl be a real or a dummy value
        Output:
              - df, filtered data frame
        OBS: tools errors should be the same and constant values.      
        '''
        a = input('a =')
        b = input('b =')
        
        df=df[(channel != -a) &  (channel != -b)]

        return df

#--------------------------------------------
    def pd2np(channel):
        '''
        Transforms a pd.DataFrame channel into an array. Type variable transformator. 
        Input:
             - df, DataFrame channel type
        Output:
             - x, array type
        '''
        x = np.array(channel)

        return x

#------------------------------------------

class Postprocessing:
    pass
    
    def err(Tcod,Pcod):
        """Função que conta os erros totais de classificação.
        Entradas:
        Tcod, vetor com os códigos de rocha verdadeiros procedentes da modelagem (True)
        Pcod, vetor com os códigos de rocha calculados pelo modelo de classificaçao (Predicted)
        Saída:
        error, contagem absoluta dos erros.
        Perror, erro em porcentagem"""
        error = 0 
        if len(Tcod) == len(Pcod):
            for i in range(len(Pcod)):
                if Tcod[i] != Pcod[i]:
                    error = error +1
                else:
                    error = error
        else:
            sys.exit("Os vetores verdadeiro e predito devem ter a mesma dimensão!")
        
        Perror = 100*error/len(Pcod)
        return error, Perror
            