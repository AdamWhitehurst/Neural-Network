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
typedef vector<double> InputVals;
typedef vector<double> TargetVals;

struct TrainingData {
public:
	double errorMargin = 0.01;
	unsigned trainingPass = 0;
	vector<InputVals> inputs;
	TargetVals targets;
	vector<double> results;
};

class NeuralNet
{
public:
	NeuralNet(const vector<unsigned> &topology);
	~NeuralNet();
	void train(TrainingData &trainingData);
	void feedForward(const vector<double> &inputs);
	void backProp(const vector<double> &targets);
	void getResults(vector<double> &results) const;
	double getRecentAverageError();
private:
	vector<Layer> _layers; // layers[layerIndex][neuronIndex]
	double _netError;
	double _recentAverageError;
	static double _recentAverageSmoothingFactor;
};

#endif // !NEURALNET_H