import { Neuron } from './neuron';

export class NeuralSense {
	private inputNeurons: Layer = [];
	private hiddenLayers: Array<Layer> = [];
	private outputNeuron = new Neuron();

	private error = 0;
	private expected = 0;

	private networkSettings: INetworkSettings;
	private trainingSettings: ITrainingSettings | null = null;

	private datasetRange: Array<number> = [];

	constructor(settings: INetworkSettings) {
		this.networkSettings = settings;

		this.initLayers();
		this.connect();
	}

	public get Error() {
		return this.error;
	}

	private initLayers() {
		const { networkSettings } = this;

		// Initializing input neurons
		for (let i = 0; i < networkSettings.inputNeurons; i++) {
			this.inputNeurons[i] = new Neuron();
		}

		// Adding a bias neuron
		if (this.networkSettings.bias) {
			this.inputNeurons.push(new Neuron(true));
		}

		// Initializing hidden layers
		for (let iLayer = 0; iLayer < networkSettings.hiddenLayers.length; ++iLayer) {
			this.hiddenLayers.push([]);

			for (let iNeuron = 0; iNeuron < networkSettings.hiddenLayers[iLayer]; ++iNeuron) {
				this.hiddenLayers[iLayer][iNeuron] = new Neuron();
			}

			if (this.networkSettings.bias) {
				this.hiddenLayers[iLayer].push(new Neuron(true));
			}
		}
	}

	private connect() {
		const { hiddenLayers } = this;

		// Connecting input neurons with neurons of the first hidden layer
		hiddenLayers[0].forEach(
			(hiddenNeuron) => {
				this.inputNeurons.forEach(
					(inputNeuron) => {
						if (hiddenNeuron?.Bias) { return; }

						inputNeuron.ConnectTo(hiddenNeuron);
					},
				);
			},
		);

		// Connecting hidden layers
		for (let iLayer = 0; iLayer < hiddenLayers.length - 1; ++iLayer) {
			for (let iCurrentNeuron = 0; iCurrentNeuron < hiddenLayers[iLayer].length; ++iCurrentNeuron) {
				const currentNeuron = hiddenLayers[iLayer][iCurrentNeuron];

				for (let iNextNeuron = 0; iNextNeuron < hiddenLayers[iLayer + 1].length; ++iNextNeuron) {
					const nextNeuron = hiddenLayers[iLayer + 1][iNextNeuron];

					if (nextNeuron?.Bias) {
						continue;
					}

					currentNeuron.ConnectTo(nextNeuron);
				}
			}
		}

		// Connecting the last neurons of the hidden layer to the output neurons
		hiddenLayers[hiddenLayers.length - 1].forEach(
			(hiddenNeuron) => {
				hiddenNeuron.ConnectTo(this.outputNeuron);
			},
		);
	}

	private normalize(task: Array<number>): Array<number> {
		let result: Array<number> = [];

		for (let i = 0; i < task.length; ++i) {
			result[i] = task[i] / this.datasetRange[i];
		}

		return result;
	}

	public denormalize(value: number) {
		return value * this.datasetRange[this.datasetRange.length - 1];
	}

	private normalizeDataset() {
		const { datasetRange } = this;

		let dataset = this.trainingSettings?.dataset;
		if (!dataset) {
			return;
		}

		for (let i = 0; i < dataset[0].length; ++i) {
			datasetRange[i] = 0;
		}

		for (const task of dataset) {
			for (let i = 0; i < task.length; ++i) {
				datasetRange[i] = datasetRange[i] < task[i] ? task[i] : datasetRange[i];
			}
		}

		for (let i = 0; i < datasetRange.length; ++i) {
			datasetRange[i] = Math.pow(10, (datasetRange[i] | 0).toString().length);
		}

		for (let i = 0; i < dataset.length; ++i) {
			for (let j = 0; j < dataset[i].length; ++j) {
				dataset[i][j] = dataset[i][j] / datasetRange[j];
			}
		}
	}

	public SetTask(task: Array<number>) {
		let inputNeuronsCount = this.inputNeurons.length;
		if (this.networkSettings.bias) {
			--inputNeuronsCount;
		}

		for (let i = 0; i < inputNeuronsCount; ++i) {
			this.inputNeurons[i].Value = task[i];
		}

		this.expected = task[task.length - 1];
	}

	public FeedForward() {
		this.hiddenLayers.forEach(
			(hiddenLayer) => {
				hiddenLayer.forEach(
					(hiddenNeuron) => hiddenNeuron.FeedForward(),
				);
			},
		);

		this.outputNeuron.FeedForward();
	}

	public BackPropagation() {
		const { outputNeuron } = this;

		this.error = outputNeuron.Value - this.expected;

		outputNeuron.Delta = this.Error * outputNeuron.GetDerivative();
		outputNeuron.BackPropagation();

		this.hiddenLayers.forEach(
			(hiddenLayer) => {
				hiddenLayer.forEach(
					(hiddenNeuron) => hiddenNeuron.BackPropagation(),
				);
			},
		);
	}

	public UpdateWeights() {
		this.outputNeuron.UpdateWeights();

		this.hiddenLayers.forEach(
			(hiddenLayer) => {
				hiddenLayer.forEach(
					(hiddenNeuron) => hiddenNeuron.UpdateWeights(),
				);
			},
		);
	}

	private initAdam() {
		this.outputNeuron.InitAdam();

		this.hiddenLayers.forEach(
			(hiddenLayer) => {
				hiddenLayer.forEach(
					(hiddenNeuron) => hiddenNeuron.InitAdam(),
				);
			},
		);
	}

	public shuffleDataset() {
		const dataset = this.trainingSettings?.dataset ?? [];

		for (let i = dataset.length - 1; i > 0; i--) {
			let j = Math.floor(Math.random() * (i + 1));
			[dataset[i], dataset[j]] = [dataset[j], dataset[i]];
		}
	}

	public Train(settings: ITrainingSettings) {
		const { optimizer } = settings.hyperParams;

		this.trainingSettings = settings;
		Neuron.HyperParams = settings.hyperParams;

		this.normalizeDataset();

		if (optimizer === 'Adam' || optimizer === 'AMSGrad') {
			this.initAdam();
		}

		let correct: number;
		for (let iEpoch = 1; iEpoch < settings.epochs; ++iEpoch) {
			this.shuffleDataset();
			correct = 0;

			for (const task of settings.dataset) {
				this.SetTask(task);

				this.FeedForward();
				this.BackPropagation();
				this.UpdateWeights();

				const isCorrect = -settings.error.threshold <= this.Error && this.Error <= settings.error.threshold;

				if (isCorrect) {
					++correct;
				}
			}

			const correctPredicts = correct / settings.dataset.length;

			if (settings.logs) {
				if (iEpoch % (settings?.logs?.period ?? 1) === 0) {
					console.log(`[Epoch ${iEpoch}]: Correct Predicts — ${correctPredicts * 100 | 0}%`);
				}
			}

			if (correctPredicts >= settings.error.correctPredicts) {
				return;
			}
		}
	}

	public Predict(task: Array<number>, log?: boolean): void | number {
		const normalize = this.normalize(task);

		this.SetTask(normalize);
		this.FeedForward();

		const output = this.outputNeuron.Value;
		const result = this.denormalize(output);

		if (!log) {
			return result;
		}

		console.log(result);
	}
}