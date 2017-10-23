using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralNetworks;
using System.Drawing;
using MathNet.Numerics.LinearAlgebra;
using System.Text.RegularExpressions;
using System.Threading;


namespace ImageRecognitionNN
{
    class Program
    {
        private static double _correct = 0;
        private static double _total = 0;
        private static Mutex _console = new Mutex();

        public static void ClearCurrentConsoleLine()
        {
            int currentLineCursor = Console.CursorTop;
            Console.SetCursorPosition(0, Console.CursorTop);
            Console.Write(new string(' ', Console.WindowWidth));
            Console.SetCursorPosition(0, currentLineCursor);
        }

        static void Main(string[] args)
        {
            Network network = new Network(784, 16, 2, 10);

            Console.WriteLine(String.Format("Testing Neural Network with {0} nodes and {1} Nueral Connections", network.Nodes, network.NeuralConnections));

            String folderPath = @"H:\NN Image Recognition Training Data\Decoded Data";

            List<String> files = new List<String>(Directory.GetFiles(folderPath, "*.bmp"));

            int batchSize = files.Count;

            for (int x = 0; x < 10; x++)
            {
                _correct = 0;
                _total = 0;

                List<Task> tasks = new List<Task>();

                foreach (var file in files)
                {
                    while (tasks.Count(task => !task.IsCompleted) > 4)
                    {
                        tasks.ForEach(task => task.Wait());
                    }

                    //tasks.Add( Task.Run(() => { TrainToFile(network, file, batchSize); }));

                    TrainToFile(network, file, batchSize);
                }
                
                while (tasks.Count( task => !task.IsCompleted) > 0);

                _console.WaitOne();
                ClearCurrentConsoleLine();
                Console.WriteLine(String.Format("Run {0} _ Accuracy {1}", x + 1, (_correct / _total).ToString("N2")));
                _console.ReleaseMutex();
            }
            Console.ReadKey();
        }


        public static void TrainToFile( Network network, String filePath, int batchSize)
        {

            Bitmap testImage = new Bitmap(filePath);

            Vector<Double> imageVector = Vector<Double>.Build.Dense(784);

            for (int p = 0; p < 784; p++)
                imageVector[p] = testImage.GetPixel(p % 28, (p / 28)).GetBrightness();

            // Construct output
            int answer = Convert.ToInt16(Regex.Match(filePath, @"(?<=Digit_)\d(?=\.)").ToString());
            Vector<Double> answerVector = Vector<Double>.Build.Dense(10, 0);
            answerVector[answer] = 1;

            Vector<Double> resultVector = network.Propergate(imageVector);
            int result = resultVector.MaximumIndex();

            _total++;
            if (answer == result) _correct++;

            //Console.Write(String.Format("File {0} _ Actual {1} _ Guess {2} _ Certainty {3} _ Correct {4}", total, answer, result, resultVector.Normalize(1)[result].ToString("N2"), correct));
            //Console.WriteLine(answer == result ? "" : " - Incorrect");

            if (_total % 500 == 0)
            {
                _console.WaitOne();
                ClearCurrentConsoleLine();
                Console.Write(String.Format("Line {0} of {1}", _total, batchSize));
                _console.ReleaseMutex();
            }

            // Train
            network.TrainToData(imageVector, answerVector, 0.1);
        }
    }
}
