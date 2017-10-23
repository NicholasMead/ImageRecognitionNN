using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetworks;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworks.Tests
{
    [TestClass()]
    public class MathTests
    {
        [TestMethod()]
        public void SigmoidTest_InputZero_OutputHalf()
        {
            Double d = 0;

            Assert.AreEqual(
                0.5,
                Math.Sigmoid(d)
                );
        }

        [TestMethod()]
        public void SigmoidTest_InputTwo_Output0_88()
        {
            Double d = 2;

            Assert.AreEqual(
                "0.88",
                Math.Sigmoid(d).ToString("N2")
                );
        }
    }
}