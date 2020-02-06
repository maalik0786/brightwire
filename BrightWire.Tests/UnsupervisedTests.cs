using System.Linq;
using BrightWire.TrainingData.Helper;
using NUnit.Framework;

namespace BrightWire.Tests
{
	public class UnsupervisedTests
	{
		static ILinearAlgebraProvider _lap;

		[SetUp]
		public static void Load() => _lap = BrightWireProvider.CreateLinearAlgebra(false);

		[Test]
		public static void Cleanup() => _lap.Dispose();

		[Test]
		public void TestKMeans()
		{
			var stringTableBuilder = new StringTableBuilder();
			var data = NaiveBayesTests.GetSimpleChineseSet(stringTableBuilder).
				ConvertToWeightedIndexList(false).Vectorise().
				ToDictionary(d => _lap.CreateVector(d.Data), d => d.Classification);
			var clusters = data.Select(d => d.Key).ToList().KMeans(_lap, 2);
			var clusterLabels = clusters.Select(d => d.Select(d2 => data[d2]).ToArray()).ToList();
		}

		[Test]
		public void TestNNMF()
		{
			var stringTableBuilder = new StringTableBuilder();
			var data = NaiveBayesTests.GetSimpleChineseSet(stringTableBuilder).
				ConvertToWeightedIndexList(false).Vectorise().
				ToDictionary(d => _lap.CreateVector(d.Data), d => d.Classification);
			var clusters = data.Select(d => d.Key).ToList().NNMF(_lap, 2);
			var clusterLabels = clusters.Select(d => d.Select(d2 => data[d2]).ToArray()).ToList();
		}
	}
}