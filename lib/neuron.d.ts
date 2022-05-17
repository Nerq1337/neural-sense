export declare class Neuron {
    Bias?: boolean | undefined;
    NeuronsInput: Array<Neuron>;
    NeuronsOutput: Array<Neuron>;
    WeightsInput: Array<number>;
    Value: number;
    Delta: number;
    private gradientSum;
    private t;
    private m;
    private v;
    private vHat;
    constructor(Bias?: boolean | undefined);
    private static hyperParams;
    static set HyperParams(hyperParams: IHyperParams);
    private activate;
    GetDerivative(): number;
    ConnectTo(neuron: Neuron): void;
    FeedForward(): void;
    BackPropagation(): void;
    InitAdam(): void;
    private adaGrad;
    private amsGrad;
    private adam;
    private defaultGD;
    UpdateWeights(): void;
}
//# sourceMappingURL=neuron.d.ts.map