#include "Neuron.h"

double Neuron::eta = 0.15;
double Neuron::alpha = 0.5;

Neuron::Neuron(unsigned numOutputs, unsigned neuronIndex)
{
	_index = neuronIndex;

	for (unsigned oi = 0; oi < numOutputs; oi++) {
		_outputWeights.push_back(Connection());
		_outputWeights.back().weight = randomWeight();

	}
}

Neuron::~Neuron()
{
}

double Neuron::randomWeight()
{
	return rand() / double(RAND_MAX);
}

void Neuron::feed(const Layer &prevLayer)
{
	double sum = 0.0;

	for (unsigned ni = 0; ni < prevLayer.size(); ni++) {
		sum += prevLayer[ni].getOutput() * prevLayer[ni]._outputWeights[_index].weight;
	}

	_output = Neuron::transferFunction(sum);
}

void Neuron::setOutput(const double newOutput)
{
	_output = newOutput;
}

double Neuron::getOutput() const
{
	return _output;
}

void Neuron::setOutputWeights(const vector<Connection> ow)
{
	_outputWeights = ow;
}

Connection & Neuron::getConnection(unsigned ci)
{
	return _outputWeights[ci];
}

void Neuron::calculateHiddenGradients(const Layer & nextLayer)
{
	// Calculate the sum of the derivative of weights
	double dow = sumDOW(nextLayer);
	_gradient = dow * Neuron::transferFunctionDerivative(_output);

			
}

void Neuron::calculateOutputGradients(double target)
{
	double delta = target;
	_gradient = delta * Neuron::transferFunctionDerivative(_output);
}

double Neuron::sumDOW(const Layer & nextLayer) const
{
	double sum = 0;

	// Sum the error contributions of this neuron
	// to the neurons that it feeds on the next layer
	for (unsigned ni = 0; ni < nextLayer.size() - 1; ni++) { // Exclude bias
		sum += _outputWeights[ni].weight * nextLayer[ni]._gradient;
	}

	return sum;
}

void Neuron::updateInputWeights(Layer & prevLayer)
{
	
}

double Neuron::transferFunction(const double sum)
{
	//TODO: Sigmoid func instead?
	return tanh(sum);
}

double Neuron::transferFunctionDerivative(const double sum)
{	// SAA: tanh(sum) ~~ sum
	return (1 - sum * sum);
}
