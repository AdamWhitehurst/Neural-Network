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

void Neuron::feed(const NeuronLayer &prevLayer)
{
	// Calculate the sum of the weighted outputs from the previous layer
	// Sum = sum of all neurons' outputs times that neuron's weight
	//		 from the previous layer
	double sum = 0.0;

	for (unsigned ni = 0; ni < prevLayer.size(); ni++) { // ni = neuron index
		sum += prevLayer[ni].getOutput() * prevLayer[ni]._outputWeights[_index].weight;
	}

	// Determine whether the sum passes the Transfer Threshold
	_output = Neuron::transferThresholdFunction(sum);
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

void Neuron::calculateHiddenGradients(const NeuronLayer & nextLayer)
{
	// Calculate the sum of the derivatives of the weights
	double dow = sumDerivativesOfWeights(nextLayer);
	_gradient = dow * Neuron::transferThresholdFunctionDerivative(_output);
}

void Neuron::calculateOutputGradients(double target)
{
	double delta = target - _output;
	_gradient = delta * Neuron::transferThresholdFunctionDerivative(_output);
}

double Neuron::sumDerivativesOfWeights(const NeuronLayer & nextLayer) const
{
	double sum = 0;

	// Sum the error contributions of this neuron
	// to the neurons that it feeds on the next layer, excluding bias neuron
	for (unsigned ni = 0; ni < nextLayer.size() - 1; ni++) { // ni = neuron index
		sum += _outputWeights[ni].weight * nextLayer[ni]._gradient;
	}

	return sum;
}

void Neuron::calculateInputWeights(NeuronLayer & prevLayer)
{
	for (unsigned ni = 0; ni < prevLayer.size(); ni++) { // ni = neuron index
		Neuron &neuron = prevLayer[ni];
		double oldDeltaWeight = neuron._outputWeights[_index].deltaWeight;

		// Calculate new delta weight
		double newDeltaWeight =
			// Neurons' individual output, magnified by the gradient and train rate:
			neuron.getOutput() * eta * _gradient
			// Also add momentum (a fraction of the old delta weight);
			+ alpha * oldDeltaWeight;

		// Set the new delta weight for this neuron's connection to the neuron on the next layer
		neuron._outputWeights[_index].deltaWeight = newDeltaWeight;

		// Modify the connection weight by the new delta weight
		neuron._outputWeights[_index].weight += newDeltaWeight;
	}
}

double Neuron::transferThresholdFunction(const double x)
{
	// Hyperbolic Tangent Function: a fast approximation to sigmoid function
	return tanh(x);
	
	// Sigmoid Function: the real deal
	// return 1 / (1 + exp(-x));
}

double Neuron::transferThresholdFunctionDerivative(const double x)
{	
	// Hyperbolic Tangent Derivative Function:
	// SAA: tanh(x) ~~ x
	return (1 - x * x);

	// Sigmoid Derivative Function
	// sigma = transferThresholdFunction(x);
	// return sigma(1 - sigma);
}
