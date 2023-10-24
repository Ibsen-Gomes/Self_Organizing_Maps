################################ Code Self-Organizing Maps (machine learning) ###################################
########################################### Ibsen P. S. Gomes ###################################################
############################ Observatório Nacional - Universidade Federal Fluminense ############################

# Libraries
import numpy as np
from math import sqrt
import pandas as pd
from numpy.ma.core import ceil
from scipy.spatial import distance #distance calculation
from sklearn.preprocessing import MinMaxScaler #normalisation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score #scoring
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib import animation, colors


# Normalization:

def norm_data(data):
    
    '''
    Data normalization using the "minmax" method
    
    input: 
    data = column in a database 
    
    operation:
    new_data = [data - min(data)]/[max(data) - min(data)]
    
    output:
    data_norm = normalized data
    '''
    
    data_norm = np.zeros((len(data)))
    for i in range(len(data)):
        data_norm[i] = (data[i] - min(data))/(max(data) - min(data))
        
    return data_norm


# Metrics:

def euclidian(v1, v2, check_input=True):
    
    '''
    Given two vectors and calculate the Euclidean distance between vectors
    
    input:
    v1 = 1D vector
    v2 = 1D vector
    
    operation:
    distance = sqrt(sum(v1 - v2)**2)
    
    output:
    euclidian_distance
    '''
    
    # "Assert" to ensure the entries are 1D:
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    if check_input is True:
        assert v1.ndim == 1, 'a must be a 1D' 
        assert v2.ndim == 1, 'x must be a 1D'
        
    dist = 0.0
    for i in range(len(v1)):
        dist += (v1[i] - v2[i])**2
 
    euclidian_distance = sqrt(dist)
    
    return euclidian_distance


def manhattan(v1, v2, check_input=True):
    
    '''
    Given two vectors and calculate the Manhattan distance between vectors
    
    input:
    v1 = 1D vector
    v2 = 1D vector
    
    operation:
    distance = |sum(v1 - v2)|
    
    output:
    euclidian_distance
    '''
    
    # "Assert" to ensure the entries are 1D:
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    if check_input is True:
        assert v1.ndim == 1, 'a must be a 1D' 
        assert v2.ndim == 1, 'x must be a 1D'
    
    dist = 0.0
    for i in range(len(v1)):
        dist += abs(v1[i] - v2[i])
 
    manhattan_distance = dist
    return manhattan_distance


# Best Matching Unit search:

def winning_neuron(data, t, som, metric='euclidian'):
    
    '''
    Calculates the distance between a sample and all neurons and chooses the BMU. 
    Process performed on all samples.
    
    input:
    data = train data
    t = random index of traing data
    som = grid of neurons
    metric = metrics: euclidian or manhattan to calculate distances.
    
    operation:
    distance (neuron, train data)
    the neuron is chosen according to the shortest distance criterion
    
    output:
    winner = winner neuron
    
    '''
    
    metrics = {
        'euclidian': euclidian,
        'manhattan': manhattan,
    }
    
    if metric not in metrics:
        raise ValueError("Function {} not recognized".format(metric))
    
    n_rows = som.shape[0]
    n_cols = som.shape[1]
    
    winner = [0,0]
    shortest_distance = np.sqrt(data.shape[1]) # initialise with max distance
    input_data = data[t]
    
    for row in range(n_rows):
        for col in range(n_cols):
            distance = metrics[metric](som[row][col], data[t]) # minha função!!!
            if distance < shortest_distance: 
                shortest_distance = distance
                winner = [row,col]
                
    return winner


# Learning rate functions:

def learning_linear(max_epochs, epochs, max_learning_rate):
    
    '''
    This function linearly decreases learning with the passing of training cycles. 
    Important operation for method convergence
    
    input:
    max_epochs = total epochs or training cycles
    epochs = epochs or training cycles 
    max_learning_rate = maximum learning rate
    
    operation:
    Linear decay in the training cycle
    
    output:
    learn_rate = Learning rate over time
    '''
     
    learn_rate = max_learning_rate*(1.0 - (np.float64(epochs)/max_epochs))        
    
    return learn_rate


def learning_inverse(max_epochs, epochs, max_learning_rate):
    
    '''
    This function decreases according to the inverse function the learning with 
    the passing of the training cycles. Important operation for method convergence
    
    input:
    max_epochs = total epochs or training cycles
    epochs = epochs or training cycles 
    max_learning_rate = maximum learning rate
    
    operation:
    Inverse decay in the training cycle
    
    output:
    learn_rate = Learning rate over time
    '''
    
    learn_rate =  max_learning_rate/(1.0 + np.float64(epochs)/max_epochs)
    
    return learn_rate


def learning_exponential(max_epochs, epochs, max_learning_rate):
    
    '''
    This function exponentially decreases learning with the passing of training cycles. 
    Important operation for method convergence
    
    input:
    max_epochs = total epochs or training cycles
    epochs = epochs or training cycles 
    max_learning_rate = maximum learning rate
    
    operation:
    Exponential decay in the training cycle
    
    output:
    learn_rate = Learning rate over time
    '''
    
    learn_rate  = max_learning_rate*np.exp(- np.float64(epochs)/max_epochs)
    
    return learn_rate


# Neighborhood functions:

def neighborhood_linear(max_epochs, epochs, max_distance):
    
    '''
    This function controls how the winning neuron influences its neighbors. 
    In this case, the neighborhood is influenced according to a linear function.
    
    input:
    max_epochs = total epochs or training cycles
    epochs = epochs or training cycles 
    max_distance = maximum distance of influence on neighbors
    
    operation:
    Linear decay of neighborhood size
    
    output:
    learn_rate = Learning rate over time
    '''
    
    neighbourhood_range = max_distance*(1.0 - (np.float64(epochs)/max_epochs))
    
    return neighbourhood_range


def neighborhood_gaussian(max_epochs, epochs, max_distance):
    
    '''
    This function controls how the winning neuron influences its neighbors. 
    In this case, the neighborhood is influenced according to the Gaussian function.
    
    input:
    max_epochs = total epochs or training cycles
    epochs = epochs or training cycles 
    max_distance = maximum distance of influence on neighbors
    
    operation:
    Neighborhood size decay with the Gaussian function
    
    output:
    learn_rate = Learning rate over time
    '''
    
    lmbda = max_epochs*0.5
    sigma = max_epochs*np.exp(-(np.float64(epochs)/lmbda))
    neighbourhood_range = ceil(max_distance*np.exp(- 0.5*(epochs/sigma)**2))
    
    return neighbourhood_range


# SOM Train:

def SOM_train(n_rows, n_cols, max_epochs, max_learning_rate, max_distance, 
              train_x, train_y, learn='linear', neigh='linear', metric='euclidian'):
    
    '''
    Cortex training using the learning rate, neighborhood function and winning neuron functions. 
    This function updates the weights of the winning neuron and its neighborhood through the training cycle.
    
    input: 
    n_nows = n° rows of grid
    n_cols = n° columns of grid
    max_epochs = total epochs or training cycles
    max_learning_rate = maximum learning rate 
    max_distance = maximum distance of influence on neighbors
    train_x = data with properties
    train_y = data with indices (used to form the map with indices)
    learn = type of learning rate
    neigh = type of neighborhood influence
    metric = type of distance operation
    
    operation:
    1°) Competition: find the BMU;
    2°) Cooperation: determine the neighborhood of the BMU;
    3°) Update: update of the weights of the BMU and its neighborhood.
    
    output:
    Cortex map after learning
    som = neurons with updated weghts
    label_map = map of neurons with the index
    '''
    
    learning = {
        'linear': learning_linear,
        'inverse': learning_inverse,
        'exponential': learning_exponential,
    }
    if learn not in learning:
        raise ValueError("Function {} not recognized".format(learn))
    
    neighborhood = {
        'linear': neighborhood_linear,
        'gaussian': neighborhood_gaussian,
    }
    if neigh not in neighborhood:
        raise ValueError("Function {} not recognized".format(neigh))
    
    metrics = {
        'euclidian': euclidian,
        'manhattan': manhattan,
    }
    if metric not in metrics:
        raise ValueError("Function {} not recognized".format(metric))
        
    # initialising self-organising map
    n_dims = train_x.shape[1] # numnber of dimensions in the input data
    np.random.seed(44)
    som = np.random.random_sample(size=(n_rows, n_cols, n_dims)) # map construction
    
    for epochs in range(max_epochs):

        learning_rate = learning[learn](max_epochs, epochs, max_learning_rate)
        neighbourhood_range = neighborhood[neigh](max_epochs, epochs, max_distance)

        t = np.random.randint(0,high=train_x.shape[0]) # random index of traing data
        winner = winning_neuron(train_x, t, som, metric)

        for row in range(n_rows):
            for col in range(n_cols):
                if metrics[metric]([row,col], winner) <= neighbourhood_range:
                    som[row][col] += learning_rate*(train_x[t]-som[row][col]) #update neighbour's weight

    return som


# Formation of the Cortex:

def Cortex_map(train_y, train_x, updated_som, max_epochs, metric='euclidian'):
    
    '''
    Cortex map after training
    ATTENTION: the input "updated_som" is the outputs of the function "SOM_train"
    
    input: 

    train_x = data with properties
    train_y = data with indices (used to form the map with indices)
    updated_som = SOM cortex alfer training in "SOM_train"
    max_epochs = total epochs or training cycles
    metric = type of distance operation
    
    output:
    Cortex map after learning
    '''
    
    n_rows = updated_som.shape[0]
    n_cols = updated_som.shape[1]
    #nn = np.linspace(1, n_rows, n_rows)
    nn = np.linspace(0, n_rows-1, n_rows)
        
    label_data = train_y
    map = np.empty(shape=(n_rows, n_cols), dtype=object)

    for row in range(n_rows):
        for col in range(n_cols):
                map[row][col] = [] # empty list to store the label

    for t in range(train_x.shape[0]):
            
        if (t+1) % 1000 == 0:
            print("sample data: ", t+1)
        winner = winning_neuron(train_x, t, updated_som, metric)
        map[winner[0]][winner[1]].append(label_data[t]) # label of winning neuron

    # construct label map
    label_map = np.zeros(shape=(n_rows, n_cols),dtype=np.int64)
    
    for row in range(n_rows):
        for col in range(n_cols):
            label_list = map[row][col]
            if len(label_list)==0:
                label = 2
            else:
                label = max(label_list, key=label_list.count)
            label_map[row][col] = label

    title = ('Final Cortex - Iteration ' + str(max_epochs))
    s_title = ('Number of Neurons = ' + str(len(nn)**2))
    #cmap = colors.ListedColormap(['#440154', '#472d7b', '#3b528b', '#21908c','#5dc863','#fde725']) 
    plt.imshow(label_map, cmap='viridis') #cmap)
    #plt.colorbar()
    plt.xticks(nn)
    plt.yticks(nn)
    plt.title(s_title, fontsize = 10)
    plt.suptitle(title, fontsize = 14)
    plt.grid(False)
    plt.show()
    
    return label_map


### SOM Classification:

def SOM_class(data_class, updated_som, label_map, metric='euclidian'):
    
    '''
    Alassification of the data after training the cortex in the function "SOM_train".
    ATTENTION: the inputs "updated_som" and "label_map" are the outputs of the function "SOM_train"
    
    input:
    data_class = data to be sorted
    updated_som = cortex updated after training on "SOM_train"
    label_map = cortex with neurons with indices
    metric = metric = type of distance operation
    
    operation:
    distance = (updated_som, data_class) 
    sample receives the index of the neuron with the closest weight
    
    output:
    crossplot of properties prop1 and prop2
    index = sorted data with index 
    
    '''
    
    metrics = {
        'euclidian': euclidian, 
        'manhattan': manhattan,
    }
    if metric not in metrics:
        raise ValueError("Function {} not recognized".format(metric))
    
    index = [0.0]*np.size(data_class,0)
    n_rows = updated_som.shape[0]
    n_cols = updated_som.shape[1]
    
    for t in range(len(data_class)):
        d = 1
        
        for row in range(n_rows):
            for col in range(n_cols):
                distance = metrics[metric](updated_som[row][col], data_class[t])
                if distance < d:
                    d = distance
                    index[t] = label_map[row][col]
    
    return index


# Validation by percentage of success:

def Per_count(real_target, predict_target):
    '''
    '''
    
    count = 0
    l = len(real_target)
    for i in range(l):
        if real_target[i] == predict_target[i]:
            count +=0
        else:
            count +=1
            
    percentage_count = (l-count)/l*100
    
    print(f'The value of percentage is {percentage_count:.3f}.')


# Crossplot of properties:

def Crossplot(index, data_class, prop1, prop2):
    
    '''
    Crossplot between 2 properties of the data base.
    
    input:
    index = index of neurons after training in "SOM_train"
    data_class = data to be sorted
    prop1 = some column of sort data (prop1 and 2 will be used for crossplot)
    prop2 = some column of sort data (prop1 and 2 will be used for crossplot)
    
    operation:
    
    output:
    Crossplot with final classification
    '''
    
    plt.close('all')
    plt.figure(figsize=(6.5,5))

    plt.scatter(data_class[:,prop1], data_class[:,prop2], c = index, cmap='viridis'
                , s=70, alpha=1.0) 
 
    plt.xlabel('Input x normalized', fontsize = 12)
    plt.ylabel('Input y normalized', fontsize = 12)
    plt.title('Self-Organizing Maps classification', fontsize = 12)
    plt.grid(True)

    #plt.savefig('SOM_Ibsen_rhobxgr.png', dpi=300,  bbox_inches = 'tight', transparent = True)
    plt.show()

    
################################################## END CODE #####################################################