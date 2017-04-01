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
	typedef vector<Neuron> Layer;

	Neuron(unsigned numOutputs, unsigned neuronIndex);
	~Neuron();

	static double randomWeight();
	void feed(const Layer &prevLayer);
	void setOutput(const double o);
	double getOutput () const;
	void setOutputWeights(const vector<Connection> ow);
	Connection& getConnection(unsigned ci);
	void calculateHiddenGradients(const Layer &nextLayer);
	void calculateOutputGradients(double target);
	double sumDOW(const Layer &nextLayer) const;
	void updateInputWeights(Layer &prevLayer);
private:
	static double eta; // [0.0...1.0] learning rate multiplier
	static double alpha; // [0.0...n] multiplier of last weight change (momentum)
	double _output;
	double _gradient;
	unsigned _index;
	vector<Connection> _outputWeights;

	static double transferFunction(const double sum);
	static double transferFunctionDerivative(const double sum);
};

#endif // !NEURON_H

