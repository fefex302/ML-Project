from keras import initializers 
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD


#parameters:
#seed: the random seed
#n_hidden_layers: number of hidden layers
#n_hidden_units: a list that contains the number of units for each hidden layer, this list must be the same length as the hidden layers 
#n_input_units: number of units of the input 
#n_output_units: number of units of the output
#activations: the first string defines the activation functions of the hidden layers and the second one defines the output one

def model_init(seed: int = None, n_hidden_layers: int = 1, n_hidden_units: list = [4],
                n_input_units: int = 17, n_output_units: int = 1, activations: tuple = ('relu','sigmoid'),
                optimizer: str = 'SGD', momentum: float = 0.8, lambd: float = 0.01, learning_rate: float = 0.01 ):

    #check parameters
    if n_hidden_layers != len(n_hidden_units):
        raise ValueError("error: the length of n_hidden_units must be the same as the value of n_hidden_layers")
    
    #define the weight initiation, for the monk it was [-0.25,0.25]
    initializer = initializers.Orthogonal(gain=1.0,seed=seed)
    
    #model creation
    model = Sequential()

    #k_regularizer = regularizers.l2(lambd)
    #k_regularizer = None 

    #adding the input layer
    model.add(Dense(units=n_input_units, kernel_initializer=initializer))
    #model.add(Dense(units=n_input_units, kernel_initializer=initializer, kernel_regularizer=k_regularizer))
    #model.add(Dense(units=n_input_units, kernel_regularizer=k_regularizer))

    #adding one or more hidden layers
    for i in range(n_hidden_layers):
        model.add(Dense(units=n_hidden_units[i], activation=activations[0], kernel_initializer=initializer))
        #model.add(Dense(units=n_hidden_units[i], activation=activations[0], kernel_initializer=initializer, kernel_regularizer=k_regularizer))
        #model.add(Dense(units=n_hidden_units[i], activation=activations[0], kernel_regularizer=k_regularizer))
    #adding the output layer with the appropriate number of units
    model.add(Dense(units=n_output_units, activation=activations[1]))

    #SGD optimizer config
    if(optimizer == 'SGD'):
        sgd = SGD(learning_rate=learning_rate, weight_decay=lambd, momentum=momentum) 
        #compilation of the model
        model.compile(optimizer=sgd, loss='mean_squared_error')



    return model