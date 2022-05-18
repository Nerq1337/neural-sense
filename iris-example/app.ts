import NeuralSense from '../lib/neural-sense';
import dataset from './dataset';

const networkSettings = {
	inputNeurons: 4,
	hiddenLayers: [10, 5],
	bias: true,
};

const network = new NeuralSense(networkSettings);

network.Train({
	dataset: dataset,
	hyperParams: {
		optimizer: 'Adam',
		learningRate: 0.001,
	},
	error: {
		threshold: 0.003,
		correctPredicts: 0.9,
	},
	logs: {
		period: 0,
	},
	epochs: Infinity,
});

const results = [
	network.Predict([5.1, 3.5, 1.4, 0.2]),
	network.Predict([4.9, 3.0, 1.4, 0.2]),
	network.Predict([4.7, 3.2, 1.3, 0.2]),
	network.Predict([4.6, 3.1, 1.5, 0.2]),
	network.Predict([5.7, 2.8, 4.5, 1.3]),
	network.Predict([6.3, 3.3, 4.7, 1.6]),
	network.Predict([4.9, 2.4, 3.3, 1.0]),
	network.Predict([6.6, 2.9, 4.6, 1.3]),
	network.Predict([6.0, 3.0, 4.8, 1.8]),
	network.Predict([6.9, 3.1, 5.4, 2.1]),
	network.Predict([6.7, 3.1, 5.6, 2.4]),
	network.Predict([6.9, 3.1, 5.1, 2.3]),
];

for (const result of results) {
	if (result >= 0.8 && result <= 1.2) {
		console.log('Iris-setosa');
	} else if (result >= 1.8 && result <= 2.2) {
		console.log('Iris-versicolor');
	} else if (result >= 2.8 && result <= 3.2) {
		console.log('Iris-virginica');
	} else {
		console.log('¯\\_(ツ)_/¯');
	}
}