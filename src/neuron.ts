class Neuron {
	public NeuronsInput: Array<Neuron> = [];
	public NeuronsOutput: Array<Neuron> = [];
	public WeightsInput: Array<number> = [];

	public Value = 0;
	public Delta = 0;

	// AdaGrad
	private gradientSum = 0;

	// Adam
	private t = 0;
	private m: number[] = [];
	private v: number[] = [];
	private vHat: number[] = [];

	constructor(public Bias?: boolean) {
		if (Bias) {
			this.Value = 1;
		}
	}

	private static hyperParams: IClassHyperParams = {
		beta1: 0.9,
		beta2: 0.999,
		epsilon: 1e-6,
		learningRate: 0.001,
		optimizer: 'None',
	};

	public static set HyperParams(hyperParams: IHyperParams) {
		Neuron.hyperParams = Object.assign(
			Neuron.hyperParams,
			hyperParams,
		);
	}

	private activate() {
		this.Value = 2 / (1 + Math.exp(-this.Value)) - 1;
	}

	public GetDerivative() {
		return 0.5 * (1 + this.Value) * (1 - this.Value);
	}

	public ConnectTo(neuron: Neuron) {
		const randomWeight = Math.random() * 2 - 1; // from 1 to -1

		this.NeuronsOutput.push(neuron);
		neuron.NeuronsInput.push(this);
		neuron.WeightsInput.push(randomWeight);
	}

	public FeedForward() {
		this.Value = 0;

		for (let i = 0; i < this.WeightsInput.length; ++i) {
			this.Value += this.WeightsInput[i] * this.NeuronsInput[i].Value;
		}

		this.activate();
	}

	public BackPropagation() {
		const { NeuronsInput, WeightsInput } = this;

		if (!this?.Bias) {
			if (NeuronsInput[0].NeuronsOutput.length === 0) {
				return;
			}

			if (NeuronsInput[0].NeuronsOutput.length > 1) {
				for (let i = 0; i < WeightsInput.length; ++i) {
					NeuronsInput[i].Delta += this.Delta * this.WeightsInput[i] * this.WeightsInput[i];
				}
				return;
			}
		}

		for (let i = 0; i < WeightsInput.length; ++i) {
			NeuronsInput[i].Delta = this.Delta * this.WeightsInput[i] * this.NeuronsInput[i].GetDerivative();
		}
	}

	public InitAdam() {
		for (let i = 0; i < this.WeightsInput.length; ++i) {
			this.m[i] = 0;
			this.v[i] = 0;
			this.vHat[i] = 0;
		}
	}

	private adaGrad() {
		const { learningRate, epsilon } = Neuron.hyperParams;

		for (let i = 0; i < this.WeightsInput.length; ++i) {
			const gradient = this.Delta * this.NeuronsInput[i].Value;

			this.gradientSum += gradient * gradient;

			this.WeightsInput[i] -= learningRate / (Math.sqrt(this.gradientSum) + epsilon) * gradient;
		}
	}

	private amsGrad() {
		const { beta1, beta2, learningRate, epsilon } = Neuron.hyperParams;

		for (let i = 0; i < this.WeightsInput.length; ++i) {
			const gradient = this.Delta * this.NeuronsInput[i].Value;

			this.m[i] = beta1 * this.m[i] + (1 - beta1) * gradient;
			this.v[i] = beta2 * this.v[i] + (1 - beta2) * gradient * gradient;

			this.vHat[i] = Math.max(this.v[i], this.vHat[i]);

			this.WeightsInput[i] -= learningRate / (Math.sqrt(this.vHat[i]) + epsilon) * this.m[i];
		}
	}

	private adam() {
		const { beta1, beta2, learningRate, epsilon } = Neuron.hyperParams;

		++this.t;

		for (let i = 0; i < this.WeightsInput.length; ++i) {
			const gradient = this.Delta * this.NeuronsInput[i].Value;

			this.m[i] = beta1 * this.m[i] + (1 - beta1) * gradient;
			this.v[i] = beta2 * this.v[i] + (1 - beta2) * gradient * gradient;

			let mHat = this.m[i];
			let vHat = this.v[i];

			if (this.t < 4) {
				vHat = vHat / (1 - Math.pow(beta2, this.t));
			}

			if (this.t < 500) {
				mHat = mHat / (1 - Math.pow(beta1, this.t));
			}

			this.WeightsInput[i] -= learningRate / (Math.sqrt(vHat) + epsilon) * mHat;
		}
	}

	private defaultGD() {
		const { hyperParams } = Neuron;

		for (let i = 0; i < this.WeightsInput.length; ++i) {
			const gradient = this.Delta * this.NeuronsInput[i].Value;
			this.WeightsInput[i] -= hyperParams.learningRate * gradient;
		}
	}

	public UpdateWeights() {
		const { NeuronsInput } = this;

		if (!this?.Bias) {
			if (NeuronsInput[0].NeuronsOutput.length > 1) {
				this.Delta *= this.GetDerivative();
			}
		}

		switch (Neuron.hyperParams.optimizer) {
			case 'Adam': {
				this.adam();
				break;
			}
			case 'AMSGrad': {
				this.amsGrad();
				break;
			}
			case 'AdaGrad': {
				this.adaGrad();
				break;
			}
			case 'None': {
				this.defaultGD();
				break;
			}
		}

		this.Delta = 0;
	}

}

export default Neuron;