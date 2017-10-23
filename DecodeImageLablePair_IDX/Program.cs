using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DecodeImageLablePair.IDX
{
    class Program
    {
        static void Main(string[] args)
        {
            int start = 10000;
            int count = 20000;

            String rootFolder = "H:/NN Image Recognition Training Data";

            List<int>       lables = GetLabels(rootFolder + "/train-labels.idx1-ubyte", start, count);
            List<Bitmap>    images = GetImages(rootFolder + "/train-images.idx3-ubyte", start, count);
            
            for (int x = 0; x < count; x++)
            {
                String savePath = String.Format("{0}/Decoded Data2/DigitRecognitionTestImage_{1}_Digit_{2}.bmp", rootFolder, start + x, lables[x]);
                images[x].Save(savePath, ImageFormat.Bmp);

                if (x % 100 == 0) Console.Write(String.Format("\rFile {0} of {1}", x, count));
            }
        }

        private static List<int> GetLabels(String filePath, int startLable, int lableReadCount)
        {
            List<int> lables = new List<int>();
            FileStream lablesFile = new FileStream(filePath, FileMode.Open);

            try
            {
                BinaryReader lablesReader = new BinaryReader(lablesFile);

                // Magic Number - discard
                // Number of Lables - discard
                lablesReader.ReadInt32();
                lablesReader.ReadInt32();

                // Read Lables
                for (int x = 0; x < lableReadCount; x++)
                {
                    if (x < startLable) continue;
                    else lables.Add(lablesReader.ReadByte());

                    if (x % 100 == 0) Console.Write(String.Format("\rLable {0} of {1}", x, lableReadCount));

                }
            }
            finally
            {
                lablesFile.Close();
            }

            return lables;
        }

        private static List<Bitmap> GetImages(String filePath, int startImage, int imageReadCount)
        {
            List<Bitmap> images = new List<Bitmap>();
            FileStream lablesFile = new FileStream(filePath, FileMode.Open);
            
            try
            {
                BinaryReader ImageReader = new BinaryReader(lablesFile);

                // Magic Number - discard
                // Number of Lables - discard
                ImageReader.ReadInt32();
                ImageReader.ReadInt32();
                ImageReader.ReadInt32();
                ImageReader.ReadInt32();

                // Read Lables
                for (int x = 0; x < imageReadCount; x++)
                {
                    if (x < startImage)
                    {
                        continue;
                    }
                    else
                    {
                        Bitmap bitmap = new Bitmap(28, 28);

                        for (int column = 0; column < 28; column++)
                        {
                            for (int row = 0; row < 28; row++)
                            {
                                int colour = ImageReader.ReadByte();
                                bitmap.SetPixel(row, column, Color.FromArgb(colour, colour, colour));
                            }
                        }

                        images.Add(bitmap);

                        if (x % 100 == 0) Console.Write(String.Format("\rImage {0} of {1}", x, imageReadCount));

                    }
                }
            }
            finally
            {
                lablesFile.Close();
            }

            return images;
        }
    }
}
