# -*- coding: utf-8 -*-
# Nom du script : test_lstm_main
# Auteur : Noubough Sleiman
# Date de création : 20-05-2024
# Description : Ce script crée un reseuax de neurone un seul neurone LSTM pour des données initialisées.
#               puis affiche le graphe correspondants
#
# But du code : vazlider le code C++

import numpy as np
import onnx 
import tensorflow as tf

# Importation des classes CommonLSTM et LSTM du fichier où elles sont définies
from ONNX_lstm import LSTMRNN

# Importation des modules utile pour tracer le graph
from onnx.helper import make_node, make_graph, make_model
from onnx.helper import make_tensor_value_info
from onnx import TensorProto

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


# Création d'un dictionnaire avec l'attribut 'opsets' spécifié
run_params = {'opsets': [], 'new_ops': [], 'log': []}

hidden_size = 4 #4 restriction obligatoire hidden_size = n_W et c_W=c_X
batch_size = 2
initial_h = tf.zeros((1, batch_size, hidden_size))
initial_c = tf.zeros((1, batch_size, W.shape[1])) #meme nombre de ligne de w 
B = tf.zeros( 2* hidden_size) #meme nombre de ligne de w 
P = np.array([[0.6, 0.2, 0.5, 0.9 ] , [0.5, 0.3, 0.5, 0.1 ], [0.7, 0.1,0.8, 0.5 ]])


# Créer un objet ONNX Node
onnx_node = onnx.helper.make_node(
    op_type="LSTMRNN",  # Spécifiez le type d'opérateur, dans ce cas "LSTM"
    inputs=["X", "W", "R", "B", "initial_h", "initial_c", "P", "layout"],  # Spécifiez les entrées de l'opérateurs
    outputs=["output", "final_hidden_state"],  # Spécifiez les sorties de l'opérateur
)


# Créez un objet LSTM avec l'objet ONNX Node créé
lstm_layer = LSTMRNN(onnx_node=onnx_node, run_params=run_params)

# Exécution de la couche LSTM avec les données d'entrée
output, final_hidden_state = lstm_layer._run(X=X, W=W, R=R, B=B,sequence_lens= None, initial_h=initial_h, initial_c=initial_c, P= P,  direction=None, hidden_size= hidden_size , input_forget=None,  layout=None)
# Affichage des résultats
print("Sortie de la couche LSTM :")
print(output)
print("\nEtat cache final :")
print(final_hidden_state)
print("\n Fin de code ")
exit()
