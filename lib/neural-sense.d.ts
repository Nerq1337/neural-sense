declare class NeuralSense {
    private inputNeurons;
    private hiddenLayers;
    private outputNeuron;
    private error;
    private expected;
    private networkSettings;
    private trainingSettings;
    private datasetRange;
    constructor(settings: INetworkSettings);
    get Error(): number;
    private initLayers;
    private connect;
    private normalize;
    denormalize(value: number): number;
    private normalizeDataset;
    SetTask(task: Array<number>): void;
    FeedForward(): void;
    BackPropagation(): void;
    UpdateWeights(): void;
    private initAdam;
    shuffleDataset(): void;
    Train(settings: ITrainingSettings): void;
    Predict(task: Array<number>, log?: boolean): void | number;
}
export default NeuralSense;
//# sourceMappingURL=neural-sense.d.ts.map