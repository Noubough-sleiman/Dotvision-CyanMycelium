# Rapport sur la Version v1 du Code LSTMNode : test_LSTM_main_v1.cpp et classLSTMNode.cpp

## Introduction
Ce document présente un résumé des améliorations apportées dans la nouvelle version du code pour la classe `LSTMNode`.
 Cette version suit la même structure de raisonnement que la version en Python.

1. **Amelioration dans l'intégration méthode LSTM**:
 Dans l'ancienne version, les grandes lignes de la méthode LSTM ont été validé sans suivre exactement la structure du code Python.
Par exemple, dans l'ancienne version, les matrices `W` étaient divisées en `Wi`, `Wo`, `Wc` et `Wf`, et les matrices `R` en `Ri`, `Ro`, `Rc` et `Rf` qui étaient initialement égales. Cependant,  dans le code Python, il y a un seul `W` et un seul `R`, qui sont ensuite divisés par la fonction `split` dans la LSTM.
   - Dans cette version, la fonction `split` a été ajoutée pour que les entrées dans le code C++ soient identiques à celles du code Python. 
   - De plus, après la division, une fonction `extend` a été ajoutée, car en Python, les tailles des vecteurs sont corrigées automatiquement.

2. **États initiaux et états cachés**:
   - Les états de cellule, les états cachés et les coefficients de connexion peepholes ont été intégrés.
   - Les équations ont été ordonnées de la même manière que dans le code Python.

## Exemple de Validation
Cette version de code est validé en utilisant le fichier Python de ONNX `ONNX_lstm.py` et un script de test `main_lstm_v1.py`.

## Prochaines Étapes
- Adapter ce code avec la librairie develloper par Guillaume.