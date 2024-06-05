# -*- coding: utf-8 -*-
# Nom du script : test_lstm_main
# Auteur : Noubough Sleiman
# Date de création : 16-05-2024
# Description : Ce script crée un reseuax de neurone un seul neurone LSTM pour des données initialisées.
#               puis affiche le graphe correspondants
#
# But du code : valider le code C++

import numpy as np
import onnx 
import tensorflow as tf

from ONNX_lstm import LSTMRNN

# Importation des modules utile pour tracer le graph
from onnx.helper import make_node, make_graph, make_model
from onnx.helper import make_tensor_value_info
from onnx import TensorProto


# Initialisation des données (Exemple 1)
"""
X = np.array([[[0.1, 0.2, 0.3],
              [0.4, 0.5, 0.6]]])

W = np.array([[[0.1, 0.2, 0.3],
               [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
                [0.2, 0.3, 0.4]]])

R = np.array([[[0.1, 0.2, 0.3, 0.1],
               [0.4, 0.5, 0.6 , 0.1],
                [0.7, 0.8, 0.9, 0.1],
                [0.2, 0.3, 0.4, 0.1]]])
"""
#Exemple 2
"""
np.random.seed(10)# Obligatoire pour que l'initialisation soit aléatoire mais identique pour différentes simulations.
W = np.random.rand(1, 4, 3)  # Poids de la matrice d'entrée
R = np.random.rand(1, 4, 3)  # Poids de la matrice de récurrence
"""

#Exemple 3
X = np.array([[[0.1, 0.2],
               [0.4, 0.5],
               [0.3, 0.4]]])

W = np.array([[[0.1, 0.2],
               [0.2, 0.7],
                [0.6, 0.2],
                [0.9,0.8]]])

R = np.array([[[0.1, 0.2, 0.4, 0.5],
               [0.4, 0.5, 0.6, 0.2],
               [0.1, 0.2, 0.8, 0.9],
               [0.4, 0.5, 0.1, 0.2]]])

X_concatenated = np.concatenate((X, X, X, X), axis=-1)
W_concatenated = np.concatenate((W, W, W, W), axis=1)
R_concatenated = np.concatenate((R, R, R, R), axis=1)


# Création d'un dictionnaire avec l'attribut 'opsets' spécifié
run_params = {'opsets': [], 'new_ops': [], 'log': []}
print("W_concatenated  :", W_concatenated,"shape", W_concatenated.shape)
print("R_concatenated  :", R_concatenated, "shape",R_concatenated.shape)

hidden_size = 4 #4 restriction obligatoire hidden_size = n_W et c_W=c_X
batch_size = 3
initial_h = tf.zeros((1, batch_size, hidden_size))
initial_c = tf.zeros((1, batch_size, W.shape[1])) #meme nombre de ligne de w 

# Créer un objet ONNX Node
onnx_node = onnx.helper.make_node(
    op_type="LSTMRNN",  # Spécifiez le type d'opérateur, dans ce cas "LSTM"
    inputs=["X", "W", "R", "B", "initial_h", "initial_c", "P", "layout"],  # Spécifiez les entrées de l'opérateurs
    outputs=["output", "final_hidden_state"],  # Spécifiez les sorties de l'opérateur
)


# Créez un objet LSTM avec l'objet ONNX Node créé
lstm_layer = LSTMRNN(onnx_node=onnx_node, run_params=run_params)

# Exécution de la couche LSTM avec les données d'entrée
output, final_hidden_state = lstm_layer._run(X=X, W=W_concatenated, R=R_concatenated, B=None,sequence_lens= None, initial_h=initial_h, initial_c=initial_c, P= None ,  direction=None, hidden_size= hidden_size , input_forget=None,  layout=None)
# Affichage des résultats
print("Sortie de la couche LSTM :")
print(output)
print("\nEtat cache final :")
print(final_hidden_state)
print("\n Fin de code ")
exit()
