from src.keras_net import model_init
import numpy as np
import time
import matplotlib.pyplot as plt

from src.utils import importMonkDataset,takeMonkInputDataset,takeMonkOutputDataset,convert_x

dataset_train = importMonkDataset('./MONK/monks-2.train')
dataset_test = importMonkDataset('./MONK/monks-2.test')

x_train = takeMonkInputDataset(dataset_train)
x_test = takeMonkInputDataset(dataset_test)

y_train = takeMonkOutputDataset(dataset_train)
y_test = takeMonkOutputDataset(dataset_test)

#Converto x,y del training set in numpy, successivamente faccio 1-hot della x e poi rimetto tutto in numpyarray
x_train = x_train.to_numpy()
y_train = y_train.to_numpy()
x_train = convert_x(x_train)
x_train = np.array(x_train)

#Converto x,y del test set in numpy, successivamente faccio 1-hot della x e poi rimetto tutto in numpyarray
x_test = x_test.to_numpy()
y_test = y_test.to_numpy()
x_test = convert_x(x_test)
x_test = np.array(x_test)

model = model_init()

#train model
history = model.fit(x=x_train,
          y=y_train,
          epochs=100,
          batch_size=124,
          validation_data=(x_test,y_test))

train_loss = history.history['loss']
train_accuracy = history.history['accuracy']
test_loss = history.history['val_loss']
test_accuracy = history.history['val_accuracy']

# predictions = model.evaluate(x_test,
#                              y_test,
#                              batch_size=432)

monk = 'monks2_3'

plt.plot(train_accuracy, label='Train Accuracy')
plt.plot(test_accuracy, label='Test Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('.\plots/'+ monk + '_accuracy') 

plt.clf()

plt.plot(train_loss, label='Train Loss')
plt.plot(test_loss, label='Test Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('.\plots/'+ monk + '_loss')

path = '.\plots/accuracy_loss.txt'

monk += '  accuracy: ' + str(test_accuracy[-1]) + ' loss: ' + str(test_loss[-1])

with open(path, 'a') as file:
    file.write(monk + '\n')


