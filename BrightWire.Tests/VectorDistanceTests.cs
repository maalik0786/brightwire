using System;
using System.Linq;
using BrightWire.Cuda;
using BrightWire.Tests.Helper;
using MathNet.Numerics.Distributions;
using NUnit.Framework;

namespace BrightWire.Tests
{
	[Category("Slow")]
	public class VectorDistanceTests
	{
		static ILinearAlgebraProvider _cuda;
		static ILinearAlgebraProvider _cpu;

		[SetUp]
		public static void Load()
		{
			_cuda = BrightWireGpuProvider.CreateLinearAlgebra(false);
			_cpu = BrightWireProvider.CreateLinearAlgebra(false);
		}

		[Test]
		public static void Cleanup()
		{
			_cuda.Dispose();
			_cpu.Dispose();
		}

		[Test]
		public void TestManhattanDistance()
		{
			var distribution = new Normal(0, 5);
			var vectors = Enumerable.Range(0, 10).Select(i =>
					_cpu.CreateVector(100, j => Convert.ToSingle(distribution.Sample())).AsIndexable()).
				ToList();
			var compareTo = Enumerable.Range(0, 20).Select(i =>
					_cpu.CreateVector(100, j => Convert.ToSingle(distribution.Sample())).AsIndexable()).
				ToList();
			var distances = _cpu.CalculateDistances(vectors, compareTo, DistanceMetric.Manhattan);
			var gpuVectors = vectors.Select(v => _cuda.CreateVector(v)).ToList();
			var gpuCompareTo = compareTo.Select(v => _cuda.CreateVector(v)).ToList();
			var gpuDistances =
				_cuda.CalculateDistances(gpuVectors, gpuCompareTo, DistanceMetric.Manhattan);
			FloatingPointHelper.AssertEqual(distances.AsIndexable(), gpuDistances.AsIndexable());
		}

		[Test]
		public void TestEuclideanDistance()
		{
			var distribution = new Normal(0, 5);
			var vectors = Enumerable.Range(0, 10).Select(i =>
					_cpu.CreateVector(100, j => Convert.ToSingle(distribution.Sample())).AsIndexable()).
				ToList();
			var compareTo = Enumerable.Range(0, 20).Select(i =>
					_cpu.CreateVector(100, j => Convert.ToSingle(distribution.Sample())).AsIndexable()).
				ToList();
			var distances = _cpu.CalculateDistances(vectors, compareTo, DistanceMetric.Euclidean);
			var gpuVectors = vectors.Select(v => _cuda.CreateVector(v)).ToList();
			var gpuCompareTo = compareTo.Select(v => _cuda.CreateVector(v)).ToList();
			var gpuDistances =
				_cuda.CalculateDistances(gpuVectors, gpuCompareTo, DistanceMetric.Euclidean);
			FloatingPointHelper.AssertEqual(distances.AsIndexable(), gpuDistances.AsIndexable());
		}

		[Test]
		public void TestCosineDistance()
		{
			var distribution = new Normal(0, 5);
			var vectors = Enumerable.Range(0, 10).Select(i =>
					_cpu.CreateVector(100, j => Convert.ToSingle(distribution.Sample())).AsIndexable()).
				ToList();
			var compareTo = Enumerable.Range(0, 20).Select(i =>
					_cpu.CreateVector(100, j => Convert.ToSingle(distribution.Sample())).AsIndexable()).
				ToList();
			var distances = _cpu.CalculateDistances(vectors, compareTo, DistanceMetric.Cosine);
			var gpuVectors = vectors.Select(v => _cuda.CreateVector(v)).ToList();
			var gpuCompareTo = compareTo.Select(v => _cuda.CreateVector(v)).ToList();
			var gpuDistances = _cuda.CalculateDistances(gpuVectors, gpuCompareTo, DistanceMetric.Cosine);
			FloatingPointHelper.AssertEqual(distances.AsIndexable(), gpuDistances.AsIndexable());
		}
	}
}