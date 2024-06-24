/**
 * @file main_lstm_v1.cpp
 * @brief Programme principal démontrant l'utilisation de la classe LSTMNode.
 */

#include <iostream>
#include <vector>
#include <cmath>
#include "cpp_lstm.cpp"


int main() {

    // Entrée des parametres d'exemple
    std::vector<std::vector<double>> X = {
            {0.1, 0.2, 0.3},
            {0.4, 0.5, 0.6}
    };  

    std::vector<std::vector<double>> W = {
            {0.1, 0.2, 0.3},
            {0.4, 0.5, 0.6},
            {0.7, 0.8, 0.9},
            {0.2, 0.3, 0.4}
    };  

    std::vector<std::vector<double>> R = {
            {0.1, 0.2, 0.3, 0.1},
            {0.4, 0.5, 0.6, 0.1},
            {0.7, 0.8, 0.9, 0.1},
            {0.2, 0.3, 0.4, 0.1}
    };      
    
    std::vector<std::vector<double>> P= {
            {0.6, 0.2, 0.5, 0.9},
            {0.5, 0.3, 0.5, 0.1},
            {0.7, 0.1, 0.8, 0.5}
    };  

    int batch_size = X.size();
    int hidden_size = 4;
    std::cout << "input_size = batch_size :"<<batch_size << std::endl;
    std::cout << "hidden_size :" <<hidden_size<< std::endl;


    //FIX ME
    //int seq_length = 1;
    //int feature : j'ai pas de feature dans le code a voir plus tard

    // Création d'un objet LSTMNode
    LSTMNode lstm(batch_size, hidden_size);
    std::cout << "start LSTM :"<< std::endl;

    // Forward pass à travers le LSTMNode
    auto result = lstm.run(X, W, R, P, {}, {}, {}, hidden_size, batch_size);

    // Extraction des résultats
    std::vector<std::vector<double>> output = result.first;
    std::vector<std::vector<double>> final_hidden_state = result.second;


   // Affichage de la sortie
    std::cout << "Output values:" << std::endl;
    for (size_t i = 0; i < output.size(); ++i) {
        for (size_t j = 0; j < output[i].size(); ++j) {
        std::cout << output[i][j] << " ";
        }
    std::cout << std::endl; 
    }

    std::cout << "fin de code!" << std::endl;

    return 0;
}


