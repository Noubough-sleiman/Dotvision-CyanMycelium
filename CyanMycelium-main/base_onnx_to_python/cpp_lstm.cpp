/**
 * @file classLSTMNode.hpp
 * @brief Déclaration de la classe LSTMNode.
 */

#include <iostream>
#include <vector>
#include <cmath>

/**
 * @class LSTMNode
 * @brief Classe représentant un noeud LSTM (Long Short-Term Memory).
 */
class LSTMNode {
public:
    LSTMNode(int batch_size, int hidden_size) : batch_size(batch_size), hidden_size(hidden_size) {
        //initializeWeights();
        initializeState();
    }

    /**
     * @brief Effectue un passage avant (forward pass) à travers le noeud LSTM.
     * 
     * @param input Matrice représentant les entrées.
     * @return std::vector<std::vector<double>> Matrice représentant l'état caché mis à jour.
     */
    std::vector<std::vector<double>> forward(const std::vector<std::vector<double>>& input) {
        // Calcul des portes d'entrée, de sortie et d'oubli     

        std::vector<std::vector<double>> input_gate = sigmoid(addMatrices(dotProduct(input, transposeMatrix(Wi)), dotProduct(hidden_state, transposeMatrix(Ri)))); 
        std::vector<std::vector<double>> forget_gate = sigmoid(addMatrices(dotProduct(input, transposeMatrix(Wf)), dotProduct(hidden_state, transposeMatrix(Rf))));
        std::vector<std::vector<double>> Curent_state =  tanh(addMatrices(dotProduct(input, transposeMatrix(Wc)), dotProduct(hidden_state,  transposeMatrix(Rc))));
        std::vector<std::vector<double>> Update_Cell_state = addMatrices(multiplyMatrices(forget_gate, cell_state), multiplyMatrices(input_gate, Curent_state));        
        std::vector<std::vector<double>> output_gate = sigmoid(addMatrices(dotProduct(input, transposeMatrix(Wo)), dotProduct(hidden_state, transposeMatrix(Ro)))); 

        hidden_state = multiplyMatrices(output_gate, tanh(Update_Cell_state));
        return hidden_state;
        
        
    }

    /**
     * @brief Effectue un passage avant (forward pass) à travers le noeud LSTM version ameliorer.
     * 
     * @param input Matrice représentant les entrées.
     * @return std::vector<std::vector<double>> Matrice représentant l'état caché mis à jour.
     */
    std::vector<std::vector<double>> forward(const std::vector<std::vector<double>>& input, std::vector<std::vector<double>>& W, std::vector<std::vector<double>>& R, std::vector<double> B, std::vector<std::vector<double>> initial_h, std::vector<std::vector<double>> initial_c,  std::vector<std::vector<double>> P ) {
        // Ce code traite la matrice input comme un seul bloc de données
        // Il ne divise pas X en sous-matrices ni n'itère sur chaque ligne comme dans certains cas en Python où le code tente de diviser le tableau X en sous-tableaux. 
        // Par conséquent, toutes les opérations sont faites sur la matrice entière en tant qu'unité.
 
        // Calcul des portes d'entrée, de sortie et d'oubli     
        std::vector<std::vector<double>> gates = addMatrices(dotProduct(input, transposeMatrix(W)), dotProduct(hidden_state, transposeMatrix(R))); 

  
        std::vector<std::vector<std::vector<double>>> gates_temp = {gates};
        std::vector<std::vector<std::vector<double>>> splitted = splitTensor(gates_temp, 4);
        
        std::vector<double> p_i, p_o, p_f; 
        std::tie(p_i, p_o, p_f) = splitTensor_x(P);

        std::vector<std::vector<double>> temp_pi = expand_vector_to_matrix(p_i, initial_c.size() ) ;
        std::vector<std::vector<double>> temp_pf = expand_vector_to_matrix(p_f, initial_c.size() ) ;
     
        std::vector<std::vector<double>> input_gate =  sigmoid(addMatrices(expand_matrix(splitted[0] , batch_size,  hidden_size), multiplyMatrices(temp_pi, initial_c  ) ));
        std::vector<std::vector<double>> forget_gate = sigmoid(addMatrices(expand_matrix(splitted[2] , batch_size,  hidden_size), multiplyMatrices(temp_pf, initial_c  )));
        std::vector<std::vector<double>> cell_gate = tanh(expand_matrix(splitted[3] , batch_size,  hidden_size));
        std::vector<std::vector<double>> Update_Cell_state = addMatrices(multiplyMatrices(forget_gate, cell_state), multiplyMatrices(input_gate, cell_gate));        

        std::vector<std::vector<double>> temp_po = expand_vector_to_matrix(p_o, Update_Cell_state.size() ) ; // est ce que p_o est toujours vecteur?
 
        std::vector<std::vector<double>> output_gate = sigmoid(addMatrices(expand_matrix(splitted[1] , batch_size,  hidden_size), multiplyMatrices(temp_po, Update_Cell_state)));
        hidden_state = multiplyMatrices(output_gate, tanh(Update_Cell_state));

        return hidden_state;

        
        
    }

    // Méthode run avec des paramètres par défaut
    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>
    run(const std::vector<std::vector<double>>& X,
        std::vector<std::vector<double>>& W,
        std::vector<std::vector<double>>& R,
        std::vector<std::vector<double>>& P,
        std::vector<double> B = {},
        std::vector<std::vector<double>> initial_h = {},
        std::vector<std::vector<double>> initial_c = {},
        int hidden_size = 0,
        int batch_size= 0) {
        
        
        // Traitement des valeurs par défaut si nécessaire
        if (B.empty()) {
            B = std::vector<double>(2 * 4 * this->hidden_size, 0.); // Initialisation par défaut
        }
        /*if (P.empty()) {
            P = std::vector<double>(3 * this->hidden_size, 0.); // Initialisation par défaut
        }*/
        if (initial_h.empty()) {
            initial_h = std::vector<std::vector<double>>(batch_size, std::vector<double>(this->hidden_size, 0.));
        }
        if (initial_c.empty()) {
            initial_c = std::vector<std::vector<double>>(batch_size, std::vector<double>(this->hidden_size, 0.));
        }
    
        std::vector<std::vector<double>> output = forward(X, W, R, B, initial_c, initial_c, P);

        // Implémentez ici la logique de votre LSTM
        std::vector<std::vector<double>> final_hidden_state =output; // Calculer l'état final

        // Pour l'exemple, renvoyer des matrices vides
        return std::make_pair(output, final_hidden_state);
    }

private:
    int batch_size;
    int hidden_size;
    std::vector<std::vector<double>> hidden_state, cell_state;  /**< Matrice représentant l'état caché et current. */
    std::vector<std::vector<double>> Wi, Wf, Wo, Wc; /**< Matrices de poids pour les différentes portes (input, forget, output, cell). */
    std::vector<std::vector<double>> Ri, Rf, Ro, Rc; /**< Matrices de poids récurrents pour les différentes portes (input, forget, output, cell). */

    void initializeWeights() {
        // Initialisation des poids W et R
        // Par souci de simplicité, les poids sont initialisés aléatoirement ici
        /*
        Wi = randomMatrix(hidden_size, input_size);
        Ri = randomMatrix(hidden_size, hidden_size);
        Wf = randomMatrix(hidden_size, input_size);
        Rf = randomMatrix(hidden_size, hidden_size);
        Wo = randomMatrix(hidden_size, input_size);
        Ro = randomMatrix(hidden_size, hidden_size);
        Wc = randomMatrix(hidden_size, input_size);
        Rc = randomMatrix(hidden_size, hidden_size);
        
        */
        
        std::vector<std::vector<double>>  W = {
        {0.1, 0.2},
        {0.2, 0.7},
        {0.6, 0.2},
        {0.9, 0.8},
        };

        
        std::vector<std::vector<double>> R = {
        {0.1, 0.2},
        {0.4, 0.5},
        {0.1, 0.2},
        {0.4, 0.5}};

        Wi = W;//randomMatrix(4, 3);
        Ri = R; 
        Wf = W;
        Rf = R; 
        Wo = W;
        Ro = R; 
        Wc = W; 
        Rc = R; 
    }

    /**
     * @brief Initialise l'état caché.
     */
    void initializeState() {

        hidden_state = std::vector<std::vector<double>>(batch_size, std::vector<double>(hidden_size, 0.0));
        cell_state = std::vector<std::vector<double>>(batch_size, std::vector<double>(hidden_size, 0.0));

        //Affichage de la matrice h
        std::cout << "Hidden_state Matrix : " << std::endl;
        printMatrix(hidden_state); 
  
    }

    /**
     * @brief Fonction d'activation sigmoid.
     */
    std::vector<std::vector<double>> sigmoid(const std::vector<std::vector<double>>& mat) {
        // Fonction d'activation sigmoïde
        std::vector<std::vector<double>> result(mat.size(), std::vector<double>(mat[0].size()));
        for (int i = 0; i < mat.size(); ++i) {
            for (int j = 0; j < mat[0].size(); ++j) {
                result[i][j] = 1 / (1 + exp(-mat[i][j]));
            }
        }
        return result;
    }


    /**
     * @brief Fonction d'activation tanh.
     */
    std::vector<std::vector<double>> tanh(const std::vector<std::vector<double>>& mat) {
        // Fonction d'activation tanh
        std::vector<std::vector<double>> result(mat.size(), std::vector<double>(mat[0].size()));
        for (int i = 0; i < mat.size(); ++i) {
            for (int j = 0; j < mat[0].size(); ++j) {
                result[i][j] = std::tanh(mat[i][j]);
            }
        }
        return result;
    }


    /**
     * @brief Produit matriciel (dot product) entre deux matrices.
     */
    std::vector<std::vector<double>> dotProduct(const std::vector<std::vector<double>>& matrix, const std::vector<std::vector<double>>& mat) {

        // Vérifier si les dimensions des matrices sont compatibles
        if(matrix[0].size() != mat.size()) {
            
            std::cout << "nbre de colone   = " << matrix[0].size() <<  "!= nbre de ligne  = " <<  mat.size() <<std::endl;
            std::cout << "Error: Incompatible matrix dimensions for dot product." << std::endl;
            
            
            std::cout << "matrix 1 :" << std::endl;
            std::cout << "colonne  " <<matrix[0].size()<< std::endl;
            std::cout << "ligne" <<matrix.size()<< std::endl;

            std::cout << " colonne  " <<mat[0].size()<< std::endl;
            std::cout << " ligne" <<mat.size()<< std::endl;


            for (int i = 0; i < matrix.size(); ++i) {
                for (int j = 0; j < matrix[0].size(); ++j) {
                    std::cout << matrix[i][j] << " ";
                }
                std::cout << std::endl;
            }
                
            std::cout << "matrix 2  :" << std::endl;
            for (int i = 0; i < mat.size(); ++i) {
                for (int j = 0; j < mat[0].size(); ++j) {
                    std::cout << mat[i][j] << " ";
                }
                std::cout << std::endl;
            }
            exit(-1); 
        }

        // Produit scalaire entre une matrice et une autre matrice
        std::vector<std::vector<double>> result(matrix.size(), std::vector<double>(mat[0].size(), 0.0));
        for (int i = 0; i < matrix.size(); ++i) {
            for (int j = 0; j < mat[0].size(); ++j) {
                for (int k = 0; k < matrix[0].size(); ++k) {
                    result[i][j] += matrix[i][k] * mat[k][j];
                }
            }
        }
        return result;
    }

    std::vector<std::vector<double>> addMatrices(const std::vector<std::vector<double>>& mat1, const std::vector<std::vector<double>>& mat2) {
        
        // Vérifier si les dimensions des matrices sont compatibles
        if((mat1[0].size() != mat2[0].size()) || (mat1.size() != mat2.size()) ) {
            std::cout << "nbre de ligne  = " << mat1[0].size() <<  "!= nbre de ligne  = " <<  mat2[0].size() <<std::endl;
            std::cout << "nbre de colonne  = " << mat1.size() <<  "!= nbre de colonne  = " <<  mat2.size() <<std::endl;
            std::cout << "Error: Incompatible matrix dimensions for dot product." << std::endl;
            exit(-1); 
        }
       
        // Addition de deux matrices
        std::vector<std::vector<double>> result(mat1.size(), std::vector<double>(mat1[0].size()));
        for (int i = 0; i < mat1.size(); ++i) {
            for (int j = 0; j < mat1[0].size(); ++j) {
                result[i][j] = mat1[i][j] + mat2[i][j];
            }
        }
        return result;
    }


    /**
     * @brief Multiplie élément par élément deux matrices.
     */
    std::vector<std::vector<double>> multiplyMatrices(const std::vector<std::vector<double>>& mat1, const std::vector<std::vector<double>>& mat2) {
        // Multiplication élément par élément de deux matrices
        std::vector<std::vector<double>> result(mat1.size(), std::vector<double>(mat1[0].size()));
        for (int i = 0; i < mat1.size(); ++i) {
            for (int j = 0; j < mat1[0].size(); ++j) {
                result[i][j] = mat1[i][j] * mat2[i][j];
            }
        }
        return result;
    }

    std::vector<std::vector<double>> randomMatrix(int rows, int cols) {
        // Initialisation d'une matrice avec des valeurs aléatoires
        std::vector<std::vector<double>> result(rows, std::vector<double>(cols));
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                result[i][j] = ((double) rand() / RAND_MAX) ; 
            }
        }
        return result;
    }


    /**
     * @brief Transpose une matrice.
     */    
    std::vector<std::vector<double>> transposeMatrix(std::vector<std::vector<double>>& matrix) {
    // Vérifier si la matrice est vide
    if(matrix.empty()){
        std::cout << " Erorr: matrix is empty" << std::endl;
        return {};
    }

    // Obtenir les dimensions de la matrice
    int rows = matrix.size();
    int cols = matrix[0].size();
    
    // Créer une nouvelle matrice pour stocker la transposée
    std::vector<std::vector<double>> transposed(cols, std::vector<double>(rows));

    // Remplir la matrice transposée
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            transposed[j][i] = matrix[i][j];
        }
    }

    return transposed;
    }


    // Fonction pour diviser le tensor en 4 parties égales le long de la dernière dimension = equivalent a split du python
    std::vector<std::vector<std::vector<double>>> splitTensor(const std::vector<std::vector<std::vector<double>>>& gates, int n) {
        int rows = gates.size();
        int cols = gates[0].size();
        int channels = gates[0][0].size();

        std::vector<std::vector<std::vector<double>>> result(n, std::vector<std::vector<double>>(rows, std::vector<double>(cols)));

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                for (int k = 0; k < channels; ++k) {
                    result[k][i][j] = gates[i][j][k];
                }
            }
        }

        return result;
    }


    // Fonction pour adapter les dimensions de la matrice `i` pour qu'elles correspondent à celles de `p_i_C_t`
    std::vector<std::vector<double>> expand_matrix(std::vector<std::vector<double>>& i, int batch_size, int hidden_size) {
        size_t new_rows = batch_size;//p_i_C_t.size();
        size_t new_cols = hidden_size; //p_i_C_t[0].size();

        std::vector<std::vector<double>> expanded_i(new_rows, std::vector<double>(new_cols, 0.0));

        for (size_t row = 0; row < new_rows; ++row) {
            for (size_t col = 0; col < new_cols; ++col) {
                 expanded_i[row][col] = i[row % i.size()][row / i.size()];
            }
        }
        return expanded_i;
    }

    void printMatrix(const std::vector<std::vector<double>>& matrix) {
        for (size_t i = 0; i < matrix.size(); ++i) {
            for (size_t j = 0; j < matrix[i].size(); ++j) {
                std::cout << matrix[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }


    std::tuple<std::vector<double>, std::vector<double>, std::vector<double>> splitTensor_x(const std::vector<std::vector<double>>& P) {
        int rows = P.size();
        if (rows != 3) {
            throw std::invalid_argument("Le nombre de lignes doit être égal à 3");
        }

        std::vector<double> p_i = P[0];
        std::vector<double> p_o = P[1];
        std::vector<double> p_f = P[2];

        return std::make_tuple(p_i, p_o, p_f);
    }

    std::vector<std::vector<double>> expand_vector_to_matrix(const std::vector<double>& vec, int duplication_factor) {
    std::vector<std::vector<double>> expanded_matrix;

    // Vérifier que le facteur de duplication est valide
    if (duplication_factor <= 0) {
        throw std::invalid_argument("Le facteur de duplication doit être supérieur à 0.");
    }

    // Dupliquer le vecteur pour créer les lignes de la matrice
    for (int i = 0; i < duplication_factor; ++i) {
        expanded_matrix.push_back(vec);
    }

    return expanded_matrix;
}

};

