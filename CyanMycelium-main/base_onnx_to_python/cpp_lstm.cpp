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
        initializeWeights();
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

        std::vector<std::vector<double>> input_gate = sigmoid(addMatrices(dotProduct(input, transposeMatrix(Wi)), dotProduct(state, transposeMatrix(Ri)))); 
        std::vector<std::vector<double>> forget_gate = sigmoid(addMatrices(dotProduct(input, transposeMatrix(Wf)), dotProduct(state, transposeMatrix(Rf))));
        std::vector<std::vector<double>> output_gate = sigmoid(addMatrices(dotProduct(input, transposeMatrix(Wo)), dotProduct(state, transposeMatrix(Ro)))); 

        // Calcul de l'état caché mis à jour
        std::vector<std::vector<double>> updated_state = addMatrices(multiplyMatrices(forget_gate, state), multiplyMatrices(input_gate, tanh(addMatrices(dotProduct(input, transposeMatrix(Wc)), dotProduct(state,  transposeMatrix(Rc))))));
        state = multiplyMatrices(output_gate, tanh(updated_state));
        return state;
    }

private:
    int batch_size;
    //int input_rows;
    //int input_cols;
    int hidden_size;
    std::vector<std::vector<double>> state;  /**< Matrice représentant l'état caché. */
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
        
        state = std::vector<std::vector<double>>(batch_size, std::vector<double>(hidden_size, 0.0));

        //Affichage de la matrice h
        std::cout << "state Matrix : " << std::endl;
        for (int i = 0; i < batch_size; ++i) {
            for (int j = 0; j < hidden_size; ++j) {
                std::cout << state[i][j] << " ";
            }
            std::cout << std::endl;
        }
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

};

