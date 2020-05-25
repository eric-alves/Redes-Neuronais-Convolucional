"""
    Autor: Eric dos Reis Alves
    
    O treinamento foi realizado utilizando o Google Colab.
    No final do arquivo está o código para fazer a importação das imagens guardadas no Google Drive
"""

import pandas as pd
import numpy as np
import os
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam

#add
from keras.applications import imagenet_utils
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

#importa o modelo mobilenet e descarta a última camada de 1000 neurônios.
base_model=MobileNet(weights='imagenet',include_top=False) 

x=base_model.output
x=GlobalAveragePooling2D()(x)
#adicionamos camadas densas para que o modelo possa aprender funções mais complexas e classificar para obter melhores resultados.
x=Dense(1024,activation='relu')(x) 
#camada densa 2
x=Dense(1024,activation='relu')(x) 
#camada densa 3
x=Dense(512,activation='relu')(x) 
#ultima camada com 4 neuronios usando a função de ativação Softmax
preds=Dense(3,activation='softmax')(x) 

#especifica os inputs e os outputs
model=Model(inputs=base_model.input,outputs=preds)
#agora um modelo foi criado com base em nossa arquitetura

#Definindo tocas as camadas para senrem treinadas
for layer in model.layers:
    layer.trainable=True

#incluído em nossas dependências
train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input) 

# indicando o caminho para a pasta de treinamento
train_generator=train_datagen.flow_from_directory('/content/drive/My Drive/pasta', 
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=True)

classes = train_generator.class_indices 
# Mostrar as classes que temos no nosso dataset     
print(classes)

# A função Loss será a cross entropy
# Será utilizada como métricas de avaliação a accuracy
model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])


step_size_train=train_generator.n//train_generator.batch_size
history = model.fit_generator(generator=train_generator,
                   steps_per_epoch=step_size_train,
                   epochs=7)


# Plotar o grafico & validation accuracy values
plt.plot(history.history['acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plotar o grafico & validation loss values
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Test', 'Test'], loc='upper left')
plt.show()



test_datagen=ImageDataGenerator(preprocessing_function=preprocess_input) #incluído em nossas dependências
test_generator=test_datagen.flow_from_directory('/content/drive/My Drive/pasta', # indicando o caminho para a pasta de teste
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=1,
                                                 class_mode='categorical',
                                                 shuffle=False)

step_size_test=test_generator.n//test_generator.batch_size
ac = model.evaluate_generator(generator=test_generator, steps=step_size_test, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)

print('Accuracy = ', ac)


#Montagem da matriz de confusão
Y_pred = model.predict_generator(generator=test_generator, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(test_generator.classes, y_pred))
print('Classification Report')
print(classification_report(test_generator.classes, y_pred, target_names=classes))

# Salvando o modelo treinado
model.save('/content/drive/My Drive/pasta')

# Importando as imagens utilizadas no teinamento (Salvas no Drive)
from google.colab import drive
drive.mount('/content/drive')