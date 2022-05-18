# NeuralSense

## Installation
With **NPM**:
```
npm i neural-sense
```

## Usage
```javascript
import NeuralSense from 'neural-sense';
```

## Examples
XOR:
```javascript
import NeuralSense from 'neural-sense';

const networkSettings = {
	inputNeurons: 2,
	hiddenLayers: [2],
	bias: false,
};

const trainingSettings = {
	dataset: [ // XOR
		[1, 1, 0],
		[0, 1, 1],
		[1, 0, 1],
		[0, 0, 0],
	],
	hyperParams: {
		optimizer: 'Adam',
		learningRate: 0.1,
	},
	error: {
		threshold: 0.001,
		correctPredicts: 1,
	},
	logs: {
		period: 1,
	},
	epochs: Infinity,
};

const network = new NeuralSense(networkSettings);

network.Train(trainingSettings);

network.Predict([1, 1], true);
network.Predict([0, 1], true);
network.Predict([1, 0], true);
network.Predict([0, 0], true);
```