/**
 * @file main_lstm.cpp
 * @brief Programme principal démontrant l'utilisation de la classe LSTMNode.
 */

#include <iostream>
#include <vector>
#include <cmath>
#include "cpp_lstm.cpp"


int main() {

    // Entrée des parametres d'exemple
    std::vector<std::vector<double>> X1 = {{0.1, 0.2, 0.3},
                                               {0.4, 0.5, 0.6}};

    std::vector<std::vector<double>> X = {{0.1, 0.2},
                                          {0.4, 0.5},
                                          {0.3, 0.4}}; 

    int batch_size = X.size();
    int hidden_size = 2;//4;
   
    std::cout << "input_size = batch_size :"<<batch_size << std::endl;
    std::cout << "hidden_size :" <<hidden_size<< std::endl;


    //FIX ME
    //int seq_length = 1;
    //std::vector<double> B , P, initial_c, initial_h ;

    // Création d'un objet LSTMNode
    LSTMNode lstm(batch_size, hidden_size);
    // Forward pass à travers le LSTMNode
    std::vector<std::vector<double>> output = lstm.forward(X);

   // Affichage de la sortie
    std::cout << "Output values:" << std::endl;
    for (size_t i = 0; i < output.size(); ++i) {
        for (size_t j = 0; j < output[i].size(); ++j) {
        std::cout << output[i][j] << " ";
        }
    std::cout << std::endl; 
    }

    std::cout << "fin de code, merci!" << std::endl;

    return 0;
}


