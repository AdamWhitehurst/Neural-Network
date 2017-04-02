// Neural Network Class Implementation
#include "NeuralNet.h"
#include <vector>

using namespace std;

double NeuralNet::_recentAverageSmoothingFactor = 100.0;

NeuralNet::NeuralNet(const vector<unsigned> &topology)
{
	unsigned numLayers = topology.size();
	for (unsigned li = 0; li < numLayers; li++) { // li = layer index
		_layers.push_back(Layer());

		// How many neurons will be in the next layer
		// the output layer has none because there is no next layer
		unsigned numOutputs = li == topology.size() - 1 ? 0 : topology[li + 1];

		// Fill new layer with neurons + bias neuron
		for (unsigned ni = 0; ni <= topology[li]; ni++) { // ni = neuron index
			_layers.back().push_back(Neuron(numOutputs, ni));
		}

		// Set bias neuron's value to 1
		_layers.back().back().setOutput(1.0);
	}
}

NeuralNet::~NeuralNet()
{
}

void NeuralNet::train(TrainingData & trainingData)
{
	assert(trainingData.errorMargin != 0);

	// Loop through training passes until the average error is within the margin of error
	do {
		trainingData.trainingPass++;

		// Choose a random set of inputs to train the network with
		unsigned random = rand() % trainingData.inputs.size();

		// Feed the inputs through the network
		feedForward(trainingData.inputs[random]);

		//Output information about current pass and inputs
		cout << "Pass " << trainingData.trainingPass << endl << "Inputs: ";
		for (unsigned ii = 0; ii < trainingData.inputs[random].size(); ii++) {
			cout << trainingData.inputs[random][ii] << " ";
		}
		cout << endl;

		// Get results of training pass
		getResults(trainingData.results);

		// Output training pass results
		cout << "Outputs: ";
		for (unsigned ri = 0; ri < trainingData.results.size() - 1; ri++) {
			cout << trainingData.results[ri] << " ";
		}

		// Output target values for training pass
		cout << endl << "Targets: " << trainingData.targets[random] << endl;

		// Feed target values back through network
		backProp(vector<double>({ trainingData.targets[random] }));

		// Output training pass's average error
		cout << "Net recent average error: "
			<< getRecentAverageError() << endl
			<< "----------------------------------" << endl;
	} while (getRecentAverageError() > trainingData.errorMargin);

	// Output success
	cout << "The Network has come within the margin for error of " << trainingData.errorMargin
		<< " in " << trainingData.trainingPass << " passes!" << endl;

	// Wait for user response
	cin.ignore();
}

void NeuralNet::feedForward(const vector<double> &inputs)
{
	// If there are less inputs arguements than
	// there are neurons in the input layer 0,
	// then we are missing information
	assert(inputs.size() == _layers[0].size() - 1);

	// Latch each input values onto the corresponding neuron
	for (unsigned ii = 0; ii < inputs.size(); ii++) { // ii = input index
		_layers[0][ii].setOutput(inputs[ii]);
	}

	// Propagate the input values through the network (ignore the input layer 0)
	for (unsigned li = 1; li < _layers.size(); li++) { // li = layer index
		// Get reference to previous layer so neurons can get outputs
		Layer &prevLayer = _layers[li - 1];

		// Feed the previous layer to each neuron on the current layer
		// Exclude the bias neuron from forward propagation
		for (unsigned ni = 0; ni < _layers[li].size() - 1; ni++) {
			_layers[li][ni].feed(prevLayer);
		}
	}
}

void NeuralNet::backProp(const vector<double> &targets)
{
	// Calculate the net Error (Using Root Mean Square)
	Layer &outputLayer = _layers.back();

	// Reset net error
	_netError = 0.0;

	// For each output neuron, excluding bias neuron
	for (unsigned ni = 0; ni < outputLayer.size() - 1; ni++) { // ni = neuron index
		// Calculate the difference between target and actual
		double delta = targets[ni] - outputLayer[ni].getOutput();

		// Square the difference
		_netError += delta*delta;
	}
	// Divide the sum of squares by the number of squares
	_netError /= outputLayer.size() - 1;

	// Take the root of that and you have the RMS
	_netError = sqrt(_netError);

	// Calculate recent average measurements
	_recentAverageError =
		(_recentAverageError * _recentAverageSmoothingFactor + _netError)
		/ (_recentAverageSmoothingFactor + 1.0);

	// Calculate output layer gradient for each neuron on the output layer, excluding bias neuron
	for (unsigned ni = 0; ni < outputLayer.size() - 1; ni++) { // ni = neuron index
		outputLayer[ni].calculateOutputGradients(targets[ni]);
	}
	// Loop through each hidden layer (not input or output layers)
	for (unsigned li = _layers.size() - 2; li > 0; li--) { // li = layer index
		Layer &hiddenLayer = _layers[li];
		Layer &nextLayer = _layers[li + 1];

		// Calculate gradients on hidden layers
		for (unsigned ni = 0; ni < hiddenLayer.size(); ni++) { // ni = neuron index
			hiddenLayer[ni].calculateHiddenGradients(nextLayer);
		}
	}

	// Propagate backwards through output and hidden layers 
	for (unsigned li = _layers.size() - 1; li > 0; li--) { // li = layer index
		Layer &layer = _layers[li];
		Layer &prevLayer = _layers[li - 1];

		//  Update connection weights for each neuron on current layer
		for (unsigned ni = 0; ni < layer.size() - 1; ni++) { // ni = neuron index
			layer[ni].calculateInputWeights(prevLayer);
		}
	}
}

void NeuralNet::getResults(vector<double>& results) const
{
	// Remove old results data
	results.clear();

	// Loop through neurons in output layer
	for (unsigned ni = 0; ni < _layers.back().size(); ni++) { // ni = neuron index
		// Push the output value to the results array
		results.push_back(_layers.back()[ni].getOutput());
	}
}
double NeuralNet::getRecentAverageError()
{
	return _recentAverageError;
}

int main()
{
	vector<unsigned> topology({ 2,2,1 });
	NeuralNet net(topology);

	TrainingData XORTraining;
	XORTraining.inputs = {
		InputVals({ 0.0, 0.0 }),
		InputVals({ 0.0, 1.0 }),
		InputVals({ 1.0, 0.0 }),
		InputVals({ 1.0, 1.0 })
	};
	XORTraining.targets = {
		0.0,
		1.0,
		1.0,
		0.0
	};
	XORTraining.errorMargin = 0.0005;

	net.train(XORTraining);
	return 1;
}
