#pragma once
#ifndef NEURON_H
#define NEURON_H

#include <vector>
#include <cstdlib> // rand()
#include <cmath> // tanh()

using namespace std;

struct Connection {
	double weight;
	double deltaWeight;
};

class Neuron
{
public:
	Neuron(unsigned numOutputs, unsigned neuronIndex);
	~Neuron();
	typedef vector<Neuron> NeuronLayer;
	static double randomWeight(); // returns a random double [0.0 ... RAND_MAX]
	void feed(const NeuronLayer &prevLayer); // Updates this neuron's output besided on the previous layer's outputs
	void setOutput(const double o); // Sets the output of this neuron
	double getOutput () const; // Returns the output of this neuron
	void setOutputWeights(const vector<Connection> ow); // Sets the weights of this neurons connections to the next layer
	void calculateHiddenGradients(const NeuronLayer &nextLayer); // Assumes this neuron is on a hidden layer and updates its gradient
	void calculateOutputGradients(double target); // Assumes this neuron is on the output layer and updates its gradient
	void calculateInputWeights(NeuronLayer &prevLayer); // Calculate the new weights of the connections to this neuron from the previous layer
	double sumDerivativesOfWeights(const NeuronLayer &nextLayer) const; // Returns the sum of the derivatives of the weights
private:
	static double eta; // [0.0 ... 1.0] learning rate multiplier
	static double alpha; // [0.0 ... n] multiplier of last weight change (momentum)
	double _output; // The output value of this neuron
	double _gradient; // THe gradient value of this neuron
	unsigned _index; // The index of this neuron on the layer it occupies
	vector<Connection> _outputWeights; // The weights of this neuron's connections to the next layer
	static double transferThresholdFunction(const double sum); 
	static double transferThresholdFunctionDerivative(const double sum);
};

#endif // !NEURON_H

