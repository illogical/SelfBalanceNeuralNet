using System.Collections.Generic;
using UnityEngine;

public class Brain : MonoBehaviour
{
    // 12:05 of Programming an Artificial Neural Network Part 2
    private ANN ann;
    private double sumSquareError;  // how closely the model fits the data that is fed into it
    
    void Start()
    {
        // Initialize the Artificial Neural Network
        ann = new ANN(2, 1, 1, 2 , 0.8);
        List<double> result;

        for (int i = 0; i < 5000; i++)
        {
            sumSquareError = 0;
            result = Train(1, 1, 0);
            sumSquareError += Mathf.Pow((float)result[0] - 0, 2);
            result = Train(1, 0, 1);
            sumSquareError += Mathf.Pow((float)result[0] - 1, 2);
            result = Train(0, 1, 1);
            sumSquareError += Mathf.Pow((float)result[0] - 1, 2);
            result = Train(0, 0, 0);
            sumSquareError += Mathf.Pow((float)result[0] - 0, 2);
        }
        
        Debug.Log($"SSE: {sumSquareError}");

        result = Train(1, 1, 0);
        Debug.Log($" 1 1 {result[0]}");
        result = Train(1, 0, 1);
        Debug.Log($" 1 0 {result[0]}");
        result = Train(0, 1, 1);
        Debug.Log($" 0 1 {result[0]}");
        result = Train(0, 0, 0);
        Debug.Log($" 0 0 {result[0]}");
    }

    private List<double> Train(double i1, double i2, double o)
    {
        List<double> inputs = new();
        List<double> outputs = new();
        inputs.Add(i1);
        inputs.Add(i2);
        outputs.Add(o);

        return ann.Train(inputs, outputs);
    }
}
