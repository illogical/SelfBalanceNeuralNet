using System.Collections.Generic;
using UnityEngine;

public class ANN
{
    public int NumInputs;
    public int NumOutputs;
    public int NumHidden;
    public int NumNeuronsPerHidden;
    public double Alpha;  // How fast the neural network will learn and determines how much a new training sample
                          // will affect/influence the current weights/training. 1 = replace existing weights.
    private List<Layer> Layers = new();

    public ANN(int numInputs, int numOutputs, int numHidden, int numNeuronsPerHidden, double alpha)
    {
        NumInputs = numInputs;
        NumOutputs = numOutputs;
        NumHidden = numHidden;
        NumNeuronsPerHidden = numNeuronsPerHidden;
        Alpha = alpha;

        if (numHidden > 0)
        {
            // Input layer
            Layers.Add(new Layer(numNeuronsPerHidden, numInputs));
            
            // Hidden layers
            for (int i = 0; i < numHidden - 1; i++)
            {
                Layers.Add(new Layer(numNeuronsPerHidden, numNeuronsPerHidden));
            }
            
            // Output layer
            Layers.Add(new Layer(numOutputs, numNeuronsPerHidden));
        }
        else
        {
            Layers.Add(new Layer(numOutputs, numInputs));
        }
    }

    public List<double> Train(List<double> inputValues, List<double> desiredOutputs)
    {
        List<double> inputs = new();
        List<double> outputs = new();

        if (inputValues.Count != NumInputs)
        {
            Debug.LogError("Input values count does not match the number of inputs.");
            return outputs;
        }

        for (int layerIndex = 0; layerIndex < NumHidden + 1; layerIndex++)  // all hidden layers + the output layer
        {
            // set the input layer to the previous layer's outputs
            if (layerIndex > 0)
            {
                inputs = new List<double>(outputs);
            }
            outputs.Clear();

            for (int neuronIndex = 0; neuronIndex < Layers[layerIndex].NumNeurons; neuronIndex++)
            {
                double N = 0; // weight * input
                Layers[layerIndex].Neurons[neuronIndex].Inputs.Clear();

                // fill inputs with outputs from the previous layer
                for (int inputIndex = 0; inputIndex < NumInputs; layerIndex++)
                {
                    Layers[layerIndex].Neurons[neuronIndex].Inputs.Add(inputs[inputIndex]);
                    // What a typical perceptron does is multiply the weights by the inputs and sums them.
                    N += Layers[layerIndex].Neurons[neuronIndex].Weights[inputIndex] * inputs[inputIndex];
                }

                // negative bias to move the decision boundry
                N -= Layers[layerIndex].Neurons[neuronIndex].Bias;
                
                Layers[layerIndex].Neurons[neuronIndex].Output = ActivationFunction(N);
                outputs.Add(Layers[layerIndex].Neurons[neuronIndex].Output);
            }

        }

        UpdateWeights(outputs, desiredOutputs);
        
        return outputs;
    }

    private void UpdateWeights(List<double> outputs, List<double> desiredOutputs)
    {
        double error;
        
        for (int i = NumHidden; i >= 0; i--) // notice we are starting with the last layer for back-propagation
        {
            for (int j = 0; j < Layers[i].NumNeurons; j++)
            {
                // output layer
                if (i == NumHidden)
                {
                    // error gradient calculated with Delta Rule
                    // how responsible this neuron is for this error (neurons are only responsible for part of the error)
                    error = desiredOutputs[j] - outputs[j];
                    Layers[i].Neurons[j].ErrorGradient = outputs[j] * (1 - outputs[j]) * error;  
                }
                else
                {
                    Layers[i].Neurons[j].ErrorGradient =
                        Layers[i].Neurons[j].Output * (1 - Layers[i].Neurons[j].Output);
                    double errorGradientSum = 0;
                    for (int p = 0; p < Layers[i + 1].NumNeurons; p++)
                    {
                        errorGradientSum += Layers[i + 1].Neurons[p].ErrorGradient * Layers[i + 1].Neurons[p].Weights[j];
                    }
                    Layers[i].Neurons[j].ErrorGradient *= errorGradientSum;

                    for (int k = 0; k < Layers[i].Neurons[k].NumInputs; k++)
                    {
                        // output layer
                        if (i == NumHidden)
                        {
                            error = desiredOutputs[j] - outputs[j];
                            Layers[i].Neurons[j].Weights[k] += Alpha * Layers[i].Neurons[j].Inputs[k] * error;
                        }
                        else
                        {
                            // hidden layers
                            // we are using the learning rate (Alpha) to adjust the weight changes more strongly in relation to the error gradient
                            // as the learning rate decreases, the network will learn more slowly but will eventually converge to a good solution.
                            // The learning rate is a hyperparameter that determines how much the network learns in each iteration.
                            // A lower learning rate may lead to slower convergence, but it may also help the network to find a better solution.
                            // A higher learning rate may lead to faster convergence, but it may also make the network more sensitive to small changes in the input data.
                            Layers[i].Neurons[j].Weights[k] += Alpha * Layers[i].Neurons[j].Inputs[k] * Alpha *
                                                               Layers[i].Neurons[j].ErrorGradient;
                        }
                        Layers[i].Neurons[j].Bias += Alpha * -1 * Layers[i].Neurons[j].ErrorGradient;
                    }
                }
            }
        }
    }

    private double ActivationFunction(double d)
    {
        throw new System.NotImplementedException();
    }
}
