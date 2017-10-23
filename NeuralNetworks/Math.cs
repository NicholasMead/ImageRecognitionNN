using System;

namespace NeuralNetworks
{
    public static class Math
    {        
        public static Double Sigmoid(Double d)
        {
            return 1 / ( 1 + System.Math.Exp( 0 - d));
        }

        public static Double InvSigmoidPrime(Double d)
        {
            return d * (1 - d);
        }
    }   
}
