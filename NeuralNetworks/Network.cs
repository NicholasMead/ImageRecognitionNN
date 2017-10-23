using Vector = MathNet.Numerics.LinearAlgebra.Vector<double>;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworks
{
    public class Network
    {
        public List<NeuronLayer> HiddernLayers { get; } = new List<NeuronLayer>();
        public NeuronLayer LastLayer { get; }

        public int InputSize { get; }
        public int OutputSize { get; }

        public int LayerCount { get; }
        public int Nodes
        {
            get
            {
                int nodes = InputSize + OutputSize;
                if (HiddernLayers.Count > 0)
                    nodes += HiddernLayers.Count * HiddernLayers[0].LayerSize; //HiddernLayers[0].Weights.ColumnCount * HiddernLayers[0].Weights.RowCount;
                return nodes;
            }
        }
        public int NeuralConnections {
            get
            {
                int connections = LastLayer.Weights.ColumnCount * LastLayer.Weights.RowCount; ;
                if (HiddernLayers.Count > 0)
                    connections += HiddernLayers.Count * HiddernLayers[0].Weights.ColumnCount * HiddernLayers[0].Weights.RowCount;
                return connections;
            }
        } 

        public Network(int inputSize, int layerSize, int layerCount, int outputSize)
        {
            InputSize = inputSize;
            OutputSize = outputSize;
            LayerCount = layerCount;

            if (layerCount > 0)
            {
                HiddernLayers.Add(new NeuronLayer(layerSize, InputSize));

                for (int x = 1; x < LayerCount; x++)
                    HiddernLayers.Add(new NeuronLayer(layerSize, layerSize));

                LastLayer = new NeuronLayer(OutputSize, layerSize);
            }
            else if (layerCount == 0)
            {
                LastLayer = new NeuronLayer(OutputSize, InputSize);
            }
            else throw new ArgumentOutOfRangeException("Input size cannot be negative");
        }

        public Vector Propergate(Vector input)
        {
            // Validate Input
            if (input.Count != InputSize) throw new ArgumentOutOfRangeException();
            
            if (LayerCount == 0) // easy case
            {
                LastLayer.Input = input;
            }
            else // move data through layers
            {
                HiddernLayers[0].Input = input;

                for (int currentLayer = 1; currentLayer < LayerCount; currentLayer++)
                    HiddernLayers[currentLayer].Input = HiddernLayers[currentLayer - 1].Output;

                LastLayer.Input = HiddernLayers.Last().Output;
            }

            return LastLayer.Output;
        }

        public double TrainToData(Vector input, Vector requiredOutput, double learningSpeed)
        {
            // Validate Input
            if (input.Count != InputSize) throw new ArgumentOutOfRangeException();

            Vector result = Propergate(input);

            LastLayer.Correction = requiredOutput - result;

            // back propergate corrections 
            if (LayerCount == 0)
            {
                // easy
            }
            else
            {
                HiddernLayers.Last().Correction = LastLayer.BackPropergation;

                for (int currectLayer = LayerCount -2; currectLayer >= 0; currectLayer--)
                {
                    HiddernLayers[currectLayer].Correction = HiddernLayers[currectLayer + 1].BackPropergation;
                }
            }

            LastLayer.CorrectForError(learningSpeed);
            HiddernLayers.ForEach(layer => layer.CorrectForError(learningSpeed));

            return (requiredOutput - result).Average(correction => System.Math.Abs(correction));
        }
    }
}
