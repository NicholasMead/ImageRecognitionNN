using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetworks;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworks.Tests
{
    [TestClass()]
    public class NeuronLayerTests
    {
        [TestMethod()]
        public void NeuronLayerTest()
        {
            NeuronLayer nl = new NeuronLayer(16, 32);

            Trace.WriteLine(nl.Weights.ToMatrixString());
        }

        [TestMethod()]
        public void PropagateTest()
        {
            Assert.Fail();
        }
    }
}