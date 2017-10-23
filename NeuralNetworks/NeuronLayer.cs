using System;

namespace NeuralNetworks
{
    using MathNet.Numerics.LinearAlgebra;
    using Vector = MathNet.Numerics.LinearAlgebra.Vector<System.Double>;
    using Matrix = MathNet.Numerics.LinearAlgebra.Matrix<System.Double>;

    public class NeuronLayer
    {
        // Layer Paramaters
        public Matrix Weights { get; set; }
        public Vector Bias { get; set; }

        // Layer Constraints
        public int LayerSize { get; }
        public int InputSize { get; }

        private Vector _input;
        private Vector _output;
        private Vector _correction;
        private Vector _delta;

        public Vector Input {
            get => _input;
            set
            {
                if (value.Count != InputSize)
                {
                    throw new ArgumentOutOfRangeException();
                }
                else
                {
                    _input = value;
                    _correction = null;

                    _output = Weights * _input;

                    _output += Bias;

                    _output = _output.Map(n => Math.Sigmoid(n), Zeros.Include);
                }
            }
        }

        public Vector Output
        {
            get
            {
                return _output;
            }
        }

        public Vector Correction
        {
            get => _correction;
            set
            {
                if (value.Count != LayerSize) throw new ArgumentException();
                _correction = value;

                _delta = Vector.Build.Dense(LayerSize);

                for (int d = 0; d < LayerSize; d++)
                {
                    _delta[d] = _correction[d] * Math.InvSigmoidPrime(_output[d]);
                }
            }
        }

        public Vector BackPropergation
        {
            get
            {
                return Weights.Transpose() * _delta;
            }
        }
        
        public NeuronLayer(int layerSize, int inputSize)
        {
            LayerSize = layerSize;
            InputSize = inputSize;

            Weights = Matrix.Build.Random(layerSize, inputSize);
            Bias = Vector.Build.Random(layerSize);
        }
        
        public void CorrectForError(double learningSpeed)
        {
            Weights += Matrix.Build.DenseOfColumnVectors(_delta) * Matrix.Build.DenseOfColumnVectors(_input).Transpose() * learningSpeed;
            Bias += _delta * learningSpeed;
        }
    }
}
