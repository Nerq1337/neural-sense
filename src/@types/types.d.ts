declare type Layer = Array<Neuron>
declare type Optimizer = 'AdaGrad' | 'Adam' | 'AMSGrad' | 'None'

declare interface INetworkSettings {
	inputNeurons: number;
	hiddenLayers: Array<number>;
	bias: boolean;
}

declare interface ITrainingSettings {
	dataset: number[][];
	epochs: number;
	logs?: {
		period: number;
	};
	error: {
		threshold: number,
		correctPredicts: number;
	};
	hyperParams: IHyperParams,
}

declare interface IHyperParams {
	optimizer: Optimizer,
	learningRate: number,
	beta1?: number,
	beta2?: number,
	epsilon?: number,
}

declare interface IClassHyperParams {
	optimizer: Optimizer,
	learningRate: number,
	beta1: number,
	beta2: number,
	epsilon: number,
}