# Self-Organizing Maps

O Self-Organizing Maps (SOM) é uma rede neural desenvolvida por KOHONEN em 1982. Esse método é inspiração no funcionamento do Córtex Cerebral, onde cada região é dedicado a uma funcionalidade do corpo humano como: visão, audição, cordenação motora, e etc. O Córtex pode ser aplicado a diferentes bancos de dados, focados em diversas áreas podendo ajudar na solução de uma infinidade de problema reais. 

O SOM é caracterizado por ser uma rede de camada única, definida ao longo de uma grade n-dimensional. A maioria das aplicações do método usam grades bidimensionais e retangulares, embora existam aplicações com grades hexagonais e espaços de uma, três ou mais dimensões. Esta rede é totalmente conectada com a camada de entrada. Isto significa que todas as n entradas estão conectadas a cada neurônio do córtex por meio de n-pesos associados. A figura abaixo mostra a arquiterura da rede SOM:

![Arquitetura do Self-Organizing Maps](https://github.com/Ibsen-Gomes/Self_Organizing_Maps/blob/main/figures/latent_space.png)

O SOM é comumente aplicado de forma não-supervisionada, porém aqui ele é aplicado de forma supervisionada. Cada amostra tem um label com valores de 0 até 5, representando um alvo. Com o avamço do treinamento da rede, os neurônios vão herdando esses labels que, posteriormente será usado para classificar um segundo banco de dados (outro poço, por exemplo). Esses labels podem representar difentes alvos dependendo da naturea do problema, neste projeto, cada label pode representar uma litologia como Folhelho, Arenito, Marga e etc.


### Treinamento do Self-Organizing Maps:

O treinamento do algoritmo Self Organization Maps (SOM) é normalmente dividido em dois estágios: competição, cooperação e adaptação. A figura abaixo mostra a dinâmica entre a rede SOM (grid em preto com neurônios nos vértices) e as amostras (nuvem roxa).

Competição: nesta etapa é selecionado o neurônio vencedor com melhor correspondência em relação aos dados de entrada, que é o neurônio destacado em "amarelo forte" na figura a;

Cooperação: a vizinhança do neurônio vencedor também é selecionado, ou seja, os neurônios dentro da região em "amarelo claro" na figura a;, também serão modificados;

Adaptação: aqui são adaptados os pesos do neurônio vencedor, bem como os dos neurônios vizinhos na rede, fazendo com que estes se aproximem da amostra (ponto branco na  figura b).

Após várias iterações, os Neurônios possuem novos pesos e a rede neural consegue representar as amostras, (figura c).

![Etapas do Self-Organizing Maps](https://github.com/Ibsen-Gomes/Self_Organizing_Maps/blob/main/figures/algorithm.png)


### Algoritmo:

O algoritmo SOM pode ser descrito nas seguintes etapas:

1. Inicialize os neurônios ($\mathbf{w}$) no espaço de parâmetros

2. Selecione uma amostra aleatória de dados ($\mathbf{x}$): $\mathbf{x}_n$

3. Encontre o neurônio mais semelhante a esta amostra, ou seja, a unidade de melhor correspondência (BMU - best-matching unit).

     $$BMU = \underset{j}{\arg \min} \left \| \mathbf{x}_n - \mathbf{w}_j \right \|$$

4. Ajuste os neurônios excitados:

     $$\Delta \mathbf{w}^{n} = \alpha(n) \cdot h(D(\mathbf{w}_{BMU},\mathbf{w}), n) \otimes (\mathbf{ x}_n - \mathbf{w}^{n})$$
     $$\mathbf{w}^{n+1} = \mathbf{w}^{n} + \Delta \mathbf{w}^{n}$$

5. Repita a partir do passo 2 até atingir o critério de parada

Onde $n$ é a época ou etapa do algoritmo, $\alpha$ é a taxa de aprendizagem que diminui a medida que passa as épocas $n$. E $h$ é a função de vizinhança, que depende da distância entre o $\mathbf{w}_{BMU}$ e os neurônios próximos $\mathbf{w}$. A função $h$ também é em função da época $n$, que tem o papel de diminuir a vizinhança quando $n$ aumenta.


### Sobre o projeto:

1. O arquivo (https://github.com/Ibsen-Gomes/Self_Organizing_Maps/blob/main/Self_Organizing_Maps.py) é responsável por armazenar todas as funções que serão utilizadas pelo método SOM. Aqui, estão presentes as equações de:
   
   1.1 taxa de aprendizado;
   
   1.2 função de vizinhança;
   
   1.3 normalização de dados;
   
   1.4 escolha do neurônio vencedor (BMU);
   
   1.5 treinamento da rede neural;
   
   1.6 plot do Cortex 2D;
   
   1.7 classificação.

2. No arquivo (https://github.com/Ibsen-Gomes/Self_Organizing_Maps/blob/main/SOM_notebook.ipynb) temos a exibição do método mostrando os resultados gráficos. Neste arquivo é possível encontrar:
   
   2.1 exibição dos dados de poço sintético usado no treinamento;
   
   2.2 separação em dados de treinamento e teste;
   
   2.3 aplicação do treinamento;
   
   2.4 porcentagem de acerto e matriz de confusão;
   
   2.5 classificação de outro poço independente mostrando crossplots e perfil geofísico.

Como resultado final, temos a classificação litológica na figura abaixo:

![Classificação litológica do Self-Organizing Maps](https://github.com/Ibsen-Gomes/Self_Organizing_Maps/blob/main/figures/perfil_classificado.png)


