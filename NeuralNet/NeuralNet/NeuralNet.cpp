// Neural Network Class Implementation
#include "NeuralNet.h"
#include <vector>

using namespace std;

NeuralNet::NeuralNet(const vector<unsigned> &topology) {
	unsigned numLayers = topology.size();
	for (unsigned li = 0; li < numLayers; li++) {
		_layers.push_back(Layer());
		unsigned numOutputs = (li == topology.size() - 1) ? 0 : topology[li + 1];

		// Fill new layer with neurons + bias neuron
		for (unsigned ni = 0; ni < topology[li] + 1; ni++) {
			_layers.back().push_back(Neuron(numOutputs, ni));
			cout << "Made neuron " << (ni + 1) << " of layer " << (li + 1) << "!\n";
		}

		// Set bias neuron's value to 1
		_layers.back().back().setOutput(1.0);
	}
}

NeuralNet::~NeuralNet()
{
}

void NeuralNet::feedForward(const vector<double> &inputs) {
	assert(inputs.size() == _layers[0].size() - 1);

	// Latch each input values onto the corresponding neuron
	for (unsigned ii = 0; ii < inputs.size(); ii++) {
		_layers[0][ii].setOutput(inputs[ii]);
	}

	// Propagate the input values through the network
	for (unsigned li = 1; li < _layers.size(); li++) {
		// Get reference to previous layer so neurons can get outputs
		Layer &prevLayer = _layers[li - 1];

		// Propagate outputs from previous layer to each neuron
		// Exclude the bias neuron from forward propagation
		for (unsigned ni = 0; ni < _layers[li].size() - 1; ni++) {
			_layers[li][ni].feed(prevLayer);
		}
	}
}

void NeuralNet::backProp(const vector<double> &targets) {
	// Calculate the net Error (Uses RMS of errors)
	Layer &outputLayer = _layers.back();
	_netError = 0.0;

	for (unsigned ni = 0; ni < outputLayer.size() - 1; ni++) { // Exclude bias neuron
		double delta = targets[ni] - outputLayer[ni].getOutput();
		_netError += delta*delta;
	}
	_netError /= outputLayer.size() - 1;
	_netError = sqrt(_netError);

	// Calculate recent average measurements
	_recentAverageError =
		(_recentAverageError * _recentAverageErrorSmoothingFactor + _netError)
		/ (_recentAverageErrorSmoothingFactor + 1.0);

	// Calculate output layer gradients
	for (unsigned ni = 0; ni < outputLayer.size() - 1; ni++) { // Exclude bias neuron
		outputLayer[ni].calculateOutputGradients(targets[ni]);
	}
	// Calculate gradients on hidden layers
	for (unsigned li = _layers.size() - 2; li > 0; li--) {
		Layer &hiddenLayer = _layers[li];
		Layer &nextLayer = _layers[li + 1];

		for (unsigned ni = 0; ni < hiddenLayer.size(); ni++) {
			hiddenLayer[ni].calculateHiddenGradients(nextLayer);
		}
	}

	// Iterate backwards through output and hidden layers to update connection weights
	for (unsigned li = _layers.size() - 1; li > 0; li--) {
		Layer &layer = _layers[li];
		Layer &prevLayer = _layers[li - 1];

		for (unsigned ni = 0; ni < layer.size(); ni++) {
			layer[ni].updateInputWeights(prevLayer);
		}
	}
}

void NeuralNet::getResults(vector<double>& results) const
{
	results.clear();
	for (unsigned ni = 0; ni < _layers.back().size(); ni++) {
		results.push_back(_layers.back()[ni].getOutput());
	}
}
/*

inputs({
{0.0, 0.0},
{1.0, 0.0},
{0.0, 1.0},
{1.0, 1.0}
});

*/
typedef vector<double> inputVals;
typedef double targetVal;
int main() {
	vector<unsigned> topology({2, 2, 1});

	vector<inputVals> inputs({
		inputVals({ 0.0, 0.0 }),
		inputVals({ 0.0, 1.0 }),
		inputVals({ 1.0, 0.0 }),
		inputVals({ 1.0, 1.0 })
	});
	vector<double> targets({
		0.0,
		1.0,
		1.0,
		0.0
	});
	vector<double> results;

	NeuralNet net(topology);

	for (unsigned i = 0; i < inputs.size(); i++) {
		net.feedForward(inputs[i]);
		cout << "Inputs: ";
		for (unsigned ii = 0; ii < inputs[i].size(); ii++) {
			cout << inputs[i][ii] << ", ";
		}
		cout << endl;

		net.getResults(results);
		cout << "Results: ";
		for (unsigned ri = 0; ri < results.size()-1; ri++) {
			cout << results[ri] << ", ";
		}
		cout << endl;
		net.backProp(vector<double>({targets[i]}));

	}

	return 1;
}