# Rapport sur la Version Initiale du Code LSTMNode : test_LSTM_main.cpp et classLSTMNode.cpp

## Introduction
Ce document présente un résumé de la première version du code pour la classe `LSTMNode`, qui implémente un nœud de mémoire à long terme (LSTM) en C++. Cette classe est utilisée pour effectuer un passage avant (forward pass) sur des données d'entrée en utilisant les mécanismes d'un LSTM.

## Description de la Classe LSTMNode

### 1. Déclaration de la Classe
La classe `LSTMNode` est définie dans le fichier `cpp_lstm`. Elle représente un nœud LSTM et inclut les méthodes et attributs nécessaires pour initialiser les poids et l'état, et pour effectuer le calcul du passage avant.

### 2. Attributs Principaux
- `batch_size` : La taille du lot (nombre d'exemples de données traités simultanément).
- `hidden_size` : La taille de l'état caché.
- `state` : Matrice représentant l'état caché actuel.
- `Wi`, `Wf`, `Wo`, `Wc` : Matrices de poids pour les portes d'entrée, de sortie, d'oubli et de cellule.
- `Ri`, `Rf`, `Ro`, `Rc` : Matrices de poids récurrents pour les portes d'entrée, de sortie, d'oubli et de cellule.

### 3. Méthodes Principales
- `LSTMNode(int batch_size, int hidden_size)` : Constructeur de la classe qui initialise les poids et l'état.
- `std::vector<std::vector<double>> forward(const std::vector<std::vector<double>>& input)` : Effectue un passage avant à travers le nœud LSTM et retourne l'état caché mis à jour.
- `void initializeWeights()` : Initialise les poids de manière aléatoire (ou avec des valeurs prédéfinies pour simplifier le code).
- `void initializeState()` : Initialise l'état caché à zéro.
- `std::vector<std::vector<double>> sigmoid(const std::vector<std::vector<double>>& mat)` : Fonction d'activation sigmoïde.
- `std::vector<std::vector<double>> tanh(const std::vector<std::vector<double>>& mat)` : Fonction d'activation tangente hyperbolique.
- `std::vector<std::vector<double>> dotProduct(const std::vector<std::vector<double>>& matrix, const std::vector<std::vector<double>>& mat)` : Produit matriciel.
- `std::vector<std::vector<double>> addMatrices(const std::vector<std::vector<double>>& mat1, const std::vector<std::vector<double>>& mat2)` : Addition de deux matrices.
- `std::vector<std::vector<double>> multiplyMatrices(const std::vector<std::vector<double>>& mat1, const std::vector<std::vector<double>>& mat2)` : Multiplication élément par élément de deux matrices.
- `std::vector<std::vector<double>> transposeMatrix(std::vector<std::vector<double>>& matrix)` : Transposition d'une matrice.

## Exemple d'Utilisation
Le fichier `main_lstm.cpp` démontre comment utiliser la classe `LSTMNode`.
Il crée un objet `LSTMNode` avec une taille de lot et une taille d'état caché définies, puis effectue un passage avant avec des données d'entrée exemple.

# Exemple de Validation
Ce code a été testé et validé en utilisant un fichier Python de ONNX `ONNX_lstm.py` et un script de test `main_lstm.py`. Les paramètres des exemples validés sont les suivants :

- Parametre 
1 :
  - `X` : Matrice d'entrée de taille (2x3).
  - `W` : Matrices de poids de taille (4x3) et (4x4) respectivement.
  - `R` : Matrices de poids de taille (4x3) et (4x4) respectivement.
  - `initial_h`: 
  - `initial_c` :

- Exemple 1 :
  - `X` = {{0.1, 0.2, 0.3},
            {0.4, 0.5, 0.6}}; Matrice d'entrée de taille (2x3).
  - `W` : std::vector<std::vector<double>> W = {
        {0.1, 0.2, 0.3},
        {0.4, 0.5, 0.6},
        {0.7, 0.8, 0.9},
        {0.2, 0.3, 0.4}}; Matrices de poids de taille (4x3) .

  - `R` :   std::vector<std::vector<double>> R = {
        {0.1, 0.2, 0.3, 0.1},
        {0.4, 0.5, 0.6, 0.1},
        {0.7, 0.8, 0.9, 0.1},
        {0.2, 0.3, 0.4, 0.1}}; Matrices de poids de taille (4x4).

  - `initial_h`:   Matrices nulle de taille (2x4).

- Exemple 2 :
  - `X` = {{0.1, 0.2},
            {0.4, 0.5}
            {0.3, 0.4}}; Matrice d'entrée de taille (3x2).

  - `W` : std::vector<std::vector<double>> W = {
        {0.1, 0.2},
        {0.2, 0.7},
        {0.6, 0.2},
        {0.9, 0.8},
        }; Matrices de poids de taille (4x2) .

  - `R` :   std::vector<std::vector<double>> R = {
        {0.1, 0.2, 0.3, 0.1},
        {0.4, 0.5, 0.6, 0.1}}; Matrices de poids de taille (4x2).
         vs      
                [0.1, 0.2, 0.4, 0.5],
               [0.4, 0.5, 0.6, 0.2],
               [0.1, 0.2, 0.8, 0.9],
               [0.4, 0.5, 0.1, 0.2]
  - `initial_h`:   Matrices nulle de taille (3x4).
  - `initial_c`:   Matrices nulle de taille (3x4).
  - `P` :  none 
  - `B` :  none  
  - `hidden_size`: 4
  - `batch_size`: 3

**Note importante :** Dans cette version actuelle du code, il est à noter que les matrices `Wi`, `Wo`, `Wc` et `Wf` sont égales, tout comme les matrices `Ri`, `Ro`, `Rc` et `Rf` sont égales. Cependant, dans le code Python correspondant, tous ces `W` sont regroupés en `W` et tous ces `R` sont regroupés en `R`. De plus, la taille des matrices `W` et `R` doit correspondre à la taille attendue par le modèle LSTM dans le code Python. . Ces différences dans la structure des paramètres doivent être améliorées dans les prochaines versions.


Il convient de noter que des restrictions peuvent s'appliquer à ces paramètres dans certains cas. 
Ces contraintes sur les paramètres sont influencées par la structure du code Python, mettant en évidence l'importance de maintenir la cohérence entre les deux implémentations. 
Par exemple, dans le code de Python nous devons respecter les conditions suivantes :

  - `Y` = [seq_length=1, num_directions=1, batch_size, hidden_size]
  - `H ` = (n_X x n_W) apres le calcul
  et puisque Y = H, donc il faut que **hidden_size = n_W**

Pour les calculs suivants de la LSTM :
  - .. = X x W.T + H x R.T + B
  - sigmoide(i + p_i * C_t)
  - C = f * C_t + i * c

Les dimensions des matrices doivent correspondre de la manière suivante :
  - .. = (n_X x c_X) x (c_W x n_W) + (c_X x hidden_size) x (c_R x n_R)      -> donc il faut que **c_X = c_W** et **c_R = hidden_size** et **n_R = n_w**
  - i = (n_X x n_W) donc pour p_i * C_t il faut que pi = (n_X x a)          -> donc C_t = (a x n_W)
  - C = (n_X x n_W)
En conclusion, les restrictions obligatoires sur les paramètres sont :
  - **hidden_size = n_W = c_R**
  - **n_R = n_W**aussi pour des assignements corrects
  - **c_W = c_X** pour un produit matriciel valide
Ainsi, il est obligatoire que **n_R = n_W = hidden_size = c_R**, aisni R doit être une matrice carrée de dimension hidden_size.

Attention : dans la version C++, ces restrictions sur la matrice R ne s'appliquent pas.


## Prochaines Étapes
Une nouvelle version du code sera développée pour inclure une fonction `run`, permettant à la structure du code de ressembler à celle du code Python. Cette version sera capable de prendre des paramètres tels que `R`, `W`, `initial_h` et `initial_c`. Si ces paramètres ne sont pas entrée, ils seront initialisés par défaut à zéro ou à des valeurs aléatoires, selon le cas.


## Prochaines Étapes
Une nouvelle version du code sera développée pour inclure une fonction `run`, permettant à la structure du code de ressembler de plus en plus à celle du code Python (à voir plus tard si `run` correspond à la fonction `activatecontext` du code de Guillaume). 
Actuellement, les paramètres  `W` et `R` sont donnés arbitrairement ou ils sont explicitement définis pour faire la validation avec la version Python. 
Dans la prochaine version, le code devra être capable de prendre en compte des paramètres tels que `R`, `W`, `initial_h` et `initial_c`. Si ces paramètres ne sont pas entrés, ils seront initialisés par défaut à zéro ou à des valeurs aléatoires, selon le cas.
