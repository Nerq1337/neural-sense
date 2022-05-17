"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.NeuralSense = void 0;
const neuron_1 = require("./neuron");
class NeuralSense {
    constructor(settings) {
        this.inputNeurons = [];
        this.hiddenLayers = [];
        this.outputNeuron = new neuron_1.Neuron();
        this.error = 0;
        this.expected = 0;
        this.trainingSettings = null;
        this.datasetRange = [];
        this.networkSettings = settings;
        this.initLayers();
        this.connect();
    }
    get Error() {
        return this.error;
    }
    initLayers() {
        const { networkSettings } = this;
        // Initializing input neurons
        for (let i = 0; i < networkSettings.inputNeurons; i++) {
            this.inputNeurons[i] = new neuron_1.Neuron();
        }
        // Adding a bias neuron
        if (this.networkSettings.bias) {
            this.inputNeurons.push(new neuron_1.Neuron(true));
        }
        // Initializing hidden layers
        for (let iLayer = 0; iLayer < networkSettings.hiddenLayers.length; ++iLayer) {
            this.hiddenLayers.push([]);
            for (let iNeuron = 0; iNeuron < networkSettings.hiddenLayers[iLayer]; ++iNeuron) {
                this.hiddenLayers[iLayer][iNeuron] = new neuron_1.Neuron();
            }
            if (this.networkSettings.bias) {
                this.hiddenLayers[iLayer].push(new neuron_1.Neuron(true));
            }
        }
    }
    connect() {
        const { hiddenLayers } = this;
        // Connecting input neurons with neurons of the first hidden layer
        hiddenLayers[0].forEach((hiddenNeuron) => {
            this.inputNeurons.forEach((inputNeuron) => {
                if (hiddenNeuron === null || hiddenNeuron === void 0 ? void 0 : hiddenNeuron.Bias) {
                    return;
                }
                inputNeuron.ConnectTo(hiddenNeuron);
            });
        });
        // Connecting hidden layers
        for (let iLayer = 0; iLayer < hiddenLayers.length - 1; ++iLayer) {
            for (let iCurrentNeuron = 0; iCurrentNeuron < hiddenLayers[iLayer].length; ++iCurrentNeuron) {
                const currentNeuron = hiddenLayers[iLayer][iCurrentNeuron];
                for (let iNextNeuron = 0; iNextNeuron < hiddenLayers[iLayer + 1].length; ++iNextNeuron) {
                    const nextNeuron = hiddenLayers[iLayer + 1][iNextNeuron];
                    if (nextNeuron === null || nextNeuron === void 0 ? void 0 : nextNeuron.Bias) {
                        continue;
                    }
                    currentNeuron.ConnectTo(nextNeuron);
                }
            }
        }
        // Connecting the last neurons of the hidden layer to the output neurons
        hiddenLayers[hiddenLayers.length - 1].forEach((hiddenNeuron) => {
            hiddenNeuron.ConnectTo(this.outputNeuron);
        });
    }
    normalize(task) {
        let result = [];
        for (let i = 0; i < task.length; ++i) {
            result[i] = task[i] / this.datasetRange[i];
        }
        return result;
    }
    denormalize(value) {
        return value * this.datasetRange[this.datasetRange.length - 1];
    }
    normalizeDataset() {
        var _a;
        const { datasetRange } = this;
        let dataset = (_a = this.trainingSettings) === null || _a === void 0 ? void 0 : _a.dataset;
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
    SetTask(task) {
        let inputNeuronsCount = this.inputNeurons.length;
        if (this.networkSettings.bias) {
            --inputNeuronsCount;
        }
        for (let i = 0; i < inputNeuronsCount; ++i) {
            this.inputNeurons[i].Value = task[i];
        }
        this.expected = task[task.length - 1];
    }
    FeedForward() {
        this.hiddenLayers.forEach((hiddenLayer) => {
            hiddenLayer.forEach((hiddenNeuron) => hiddenNeuron.FeedForward());
        });
        this.outputNeuron.FeedForward();
    }
    BackPropagation() {
        const { outputNeuron } = this;
        this.error = outputNeuron.Value - this.expected;
        outputNeuron.Delta = this.Error * outputNeuron.GetDerivative();
        outputNeuron.BackPropagation();
        this.hiddenLayers.forEach((hiddenLayer) => {
            hiddenLayer.forEach((hiddenNeuron) => hiddenNeuron.BackPropagation());
        });
    }
    UpdateWeights() {
        this.outputNeuron.UpdateWeights();
        this.hiddenLayers.forEach((hiddenLayer) => {
            hiddenLayer.forEach((hiddenNeuron) => hiddenNeuron.UpdateWeights());
        });
    }
    initAdam() {
        this.outputNeuron.InitAdam();
        this.hiddenLayers.forEach((hiddenLayer) => {
            hiddenLayer.forEach((hiddenNeuron) => hiddenNeuron.InitAdam());
        });
    }
    shuffleDataset() {
        var _a, _b;
        const dataset = (_b = (_a = this.trainingSettings) === null || _a === void 0 ? void 0 : _a.dataset) !== null && _b !== void 0 ? _b : [];
        for (let i = dataset.length - 1; i > 0; i--) {
            let j = Math.floor(Math.random() * (i + 1));
            [dataset[i], dataset[j]] = [dataset[j], dataset[i]];
        }
    }
    Train(settings) {
        var _a, _b;
        const { optimizer } = settings.hyperParams;
        this.trainingSettings = settings;
        neuron_1.Neuron.HyperParams = settings.hyperParams;
        this.normalizeDataset();
        if (optimizer === 'Adam' || optimizer === 'AMSGrad') {
            this.initAdam();
        }
        let correct;
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
                if (iEpoch % ((_b = (_a = settings === null || settings === void 0 ? void 0 : settings.logs) === null || _a === void 0 ? void 0 : _a.period) !== null && _b !== void 0 ? _b : 1) === 0) {
                    console.log(`[Epoch ${iEpoch}]: Correct Predicts — ${correctPredicts * 100 | 0}%`);
                }
            }
            if (correctPredicts >= settings.error.correctPredicts) {
                return;
            }
        }
    }
    Predict(task, log) {
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
exports.NeuralSense = NeuralSense;
//# sourceMappingURL=neural-sense.js.map