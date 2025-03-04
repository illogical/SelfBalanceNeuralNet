using System.Collections.Generic;
using UnityEngine;

public class Layer
{
    public int NumNeurons;
    public List<Neuron> Neurons = new();

    public Layer(int numNeurons, int numNeuronInputs)
    {
        NumNeurons = numNeurons;
        
        for (int i = 0; i < numNeurons; i++)
        {
            Neurons.Add(new Neuron(numNeuronInputs));
        }
    }
}
