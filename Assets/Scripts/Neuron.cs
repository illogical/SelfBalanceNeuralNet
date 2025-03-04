using System.Collections.Generic;
using UnityEngine;  

public class Neuron // a single Neuron acts as a Perceptron
{
    public int NumInputs;
    public double Bias;
    public double Output;
    public double ErrorGradient;
    public List<double> Weights = new();
    public List<double> Inputs = new();

    public Neuron(int numInputs)
    {
        NumInputs = numInputs;
        Bias = Random.Range(-1.0f, 1.0f);
        
        for (int i = 0; i < numInputs; i++)
        {
            Weights.Add(Random.Range(-1.0f, 1.0f));
        }
    }
}
