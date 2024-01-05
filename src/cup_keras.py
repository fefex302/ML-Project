import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
from itertools import product

from keras_net import model_init
from utils import importDatasetCup
from validation_functions import Kfold, split_folds, train_test_split

PATH = 'CUP\ML-CUP23-TR.csv'
K_FOLDS = 4
SEED = int(time.time()%150)

start_time = time.time()
dataset = importDatasetCup(PATH)

#we are going to do CV on training set, holding out a portion for model assessment
#seed = 42 example of "unlucky" fold
training_set, test_set = train_test_split(dataset=dataset,test_ratio=0.2, seed=SEED)

train, val = Kfold(dataset=training_set, k_fold=K_FOLDS)
#for lr,momentum in product(learning_rates,momentums):

#take the k lists containing the k folds splitted into x and y. Each of the value returned is a list
x_train,x_test,y_train,y_test = split_folds(val,train)

#HYPERPARAMETERS

# 20,20,40
# 40,40,80
# 80,80,160
# 160,160,256
# 50,100,80
# 40,80,100,80
# 64,128,200,128
# 40,40,80,100,100
#n_hunits = [[32,16,8],[64,32,16],[20,10,10],[100,50,25],[100,50]]
#n_hunits = [[32,64,128,256,256]]
n_hunits = [[500,500,800]]
#n_hunits = [[100,50],[128,64],[100,50,25]]
l_rates = [0.0007]
lambdas = [0.001]
momentums = [0.9]
batch = 64#len(train[0])


medium_loss_training = 0
medium_loss_validation = 0
palette = sns.color_palette("husl", K_FOLDS *2)

for units,lr,lambd,momentum in product(n_hunits,l_rates,lambdas,momentums):
    hist_t = []
    hist_v = []
    for i in range(K_FOLDS):

        model = model_init(n_hidden_layers=len(units),
                        n_hidden_units=units,
                        n_output_units=3,
                        n_input_units=10,
                        activations=('tanh','linear'),
                        learning_rate=lr,
                        lambd=lambd,
                        momentum=momentum,
                        seed=SEED
                        )    

        history = model.fit(x=x_train[i],
            y=y_train[i],
            epochs=2000,
            batch_size=batch,
            validation_data=(x_test[i],y_test[i]),
            verbose=1)

        
        hist_t.append(history.history['loss'])
        hist_v.append(history.history['val_loss'])
        medium_loss_training += history.history['loss'][-1]
        medium_loss_validation += history.history['val_loss'][-1]

    medium_loss_training = medium_loss_training/K_FOLDS 
    medium_loss_validation = medium_loss_validation/K_FOLDS
    path = '.\cup_plots_CV/loss_tr_val.txt'

    #tostr = 'medium_loss_training:' + ' '+ str(medium_loss_training) +  ' units: ' + str(units) + ' lr: ' + str(lr) + ' lambda: ' + str(lambd) + ' momentum: ' + str(momentum)
    tostr = f'units: {str(units)} lr: {str(lr)} lambda: {str(lambd)} momentum: {str(momentum)} medium_loss_tr: {str(medium_loss_training)} medium_loss_val: {str(medium_loss_validation)}'
    
    plt.clf()
    plt.ylim(0,5)

    mean_train_loss = np.mean(hist_t, axis=0)
    std_train_loss = np.std(hist_t, axis=0)
    mean_val_loss = np.mean(hist_v, axis=0)
    std_val_loss = np.std(hist_v, axis=0)
    plt.plot(mean_train_loss, label=f'Train Loss')
    plt.plot(mean_val_loss, label=f'Validation Loss')
    # plt.fill_between(200, mean_train_loss - std_train_loss, mean_train_loss + std_train_loss, alpha =0.2)

    # for i in range(K_FOLDS):
    #     plt.plot(hist_t[i], label=f'Train Loss {i + 1}',color=palette[i])
    # for i in range(K_FOLDS):    
    #     plt.plot(hist_v[i], label=f'Test Loss {i + 1}', linestyle='--', color=palette[i + K_FOLDS])


    # # Plottiamo l'intervallo di deviazione standard per le curve di training e validation
    # plt.fill_between(300, mean_train_loss - std_train_loss, mean_train_loss + std_train_loss, color=palette[0], alpha=0.2)
    # plt.fill_between(300, mean_val_loss - std_val_loss, mean_val_loss + std_val_loss, color=palette[len(hist_t)], alpha=0.2)

    plt.title(f'Medium Loss - {str(K_FOLDS)} Folds')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('.\cup_plots_CV/'+ str(units)+ '_' + str(lr) + '_' +str(lambd) + '_' + str(momentum) + '.png')
    with open(path, 'a') as file:
        file.write(tostr + '\n')
    # with open('.\cup_plots/predictions.txt', 'a') as file:
    #     file.write(str(predicted) + '\n'+ '\n')

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Tempo totale di esecuzione: {elapsed_time} secondi")




