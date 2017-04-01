// Neural Network Class declaration
// Defines a feedforward, backpropagating neural network

// topology = structure containing information for 
//			  constructing neural net: number of nodes, layers, etc.
#pragma once
#ifndef  NEURALNET_H
#define NEURALNET_H

#include <vector>
#include <iostream> // cout, cin
#include <cassert> // assert()
#include "Neuron.h"

using namespace std;

typedef vector<Neuron> Layer;

class NeuralNet
{
public:
	NeuralNet(const vector<unsigned> &topology);
	~NeuralNet();
	void feedForward(const vector<double> &inputs);
	void backProp(const vector<double> &targets);
	void getResults(vector<double> &results) const;
private:
	vector<Layer> _layers; // layers[layerIndex][neuronIndex]
	double _netError;
	double _recentAverageError;
	double _recentAverageErrorSmoothingFactor;
};

#endif // !NEURALNET_H