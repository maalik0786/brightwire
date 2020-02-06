using System;
using System.Collections.Generic;
using System.Linq;
using BrightWire.Cuda;
using BrightWire.Tests.Helper;
using MathNet.Numerics.Distributions;
using NUnit.Framework;

namespace BrightWire.Tests
{
	public class CudaVectorTests
	{
		static ILinearAlgebraProvider _cuda;
		static ILinearAlgebraProvider _cpu;

		private static void Load()
		{
			_cuda = BrightWireGpuProvider.CreateLinearAlgebra(false);
			_cpu = BrightWireProvider.CreateLinearAlgebra(false);
		}

		private static void Cleanup()
		{
			_cuda.Dispose();
			_cpu.Dispose();
		}

		[Test, Category("Slow")]
		public void TestVectorCreation()
		{
			Load();
			var values = Enumerable.Range(0, 10).Select(v => (float)v).ToList();
			var a = _cpu.CreateVector(values).AsIndexable();
			Assert.AreEqual(a[4], 4f);
			Assert.AreEqual(a[0], 0f);
			Assert.AreEqual(a[9], 9f);
			IIndexableVector gpuResults;
			using (var gpuA = _cuda.CreateVector(values))
				gpuResults = gpuA.AsIndexable();
			FloatingPointHelper.AssertEqual(gpuResults, a);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void VectorColumnMatrix()
		{
			Load();
			var a = _cpu.CreateVector(5, i => i).AsIndexable();
			var matrix = a.ReshapeAsColumnMatrix().AsIndexable();
			IIndexableMatrix gpuResults;
			using (var gpuA = _cuda.CreateVector(a))
			{
				using var gpuB = gpuA.ReshapeAsColumnMatrix();
				gpuResults = gpuB.AsIndexable();
			}

			FloatingPointHelper.AssertEqual(matrix, gpuResults);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void VectorRowMatrix()
		{
			Load();
			var a = _cpu.CreateVector(5, i => i).AsIndexable();
			var matrix = a.ReshapeAsRowMatrix().AsIndexable();
			IIndexableMatrix gpuResults;
			using (var gpuA = _cuda.CreateVector(a))
			{
				using var m = gpuA.ReshapeAsRowMatrix();
				gpuResults = m.AsIndexable();
			}

			FloatingPointHelper.AssertEqual(matrix, gpuResults);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void VectorAdd()
		{
			Load();
			var a = _cpu.CreateVector(5, i => i).AsIndexable();
			var b = _cpu.CreateVector(5, i => i * 2).AsIndexable();
			var c = a.Add(b).AsIndexable();
			IIndexableVector gpuResults;
			using (var gpuA = _cuda.CreateVector(a))
			{
				using var gpuB = _cuda.CreateVector(b);
				using var gpuC = gpuA.Add(gpuB);
				gpuResults = gpuC.AsIndexable();
			}

			FloatingPointHelper.AssertEqual(c, gpuResults);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void VectorSubtract()
		{
			Load();
			var a = _cpu.CreateVector(5, i => i).AsIndexable();
			var b = _cpu.CreateVector(5, i => i * 2).AsIndexable();
			var c = a.Subtract(b).AsIndexable();
			IIndexableVector gpuResults;
			using (var gpuA = _cuda.CreateVector(a))
			{
				using var gpuB = _cuda.CreateVector(b);
				using var gpuC = gpuA.Subtract(gpuB);
				gpuResults = gpuC.AsIndexable();
			}

			FloatingPointHelper.AssertEqual(c, gpuResults);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void VectorPointwiseMultiply()
		{
			Load();
			var a = _cpu.CreateVector(5, i => i).AsIndexable();
			var b = _cpu.CreateVector(5, i => i * 2).AsIndexable();
			var c = a.PointwiseMultiply(b).AsIndexable();
			IIndexableVector gpuResults;
			using (var gpuA = _cuda.CreateVector(a))
			{
				using var gpuB = _cuda.CreateVector(b);
				using var gpuC = gpuA.PointwiseMultiply(gpuB);
				gpuResults = gpuC.AsIndexable();
			}

			FloatingPointHelper.AssertEqual(c, gpuResults);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void VectorDotProduct()
		{
			Load();
			var a = _cpu.CreateVector(5, i => i).AsIndexable();
			var b = _cpu.CreateVector(5, i => i * 2).AsIndexable();
			var dot1 = a.DotProduct(b);
			float dot2;
			using (var gpuA = _cuda.CreateVector(a))
			{
				using var gpuB = _cuda.CreateVector(b);
				dot2 = gpuA.DotProduct(gpuB);
			}

			Assert.AreEqual(dot1, dot2);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void VectorL2Norm()
		{
			Load();
			var a = _cpu.CreateVector(5, i => i).AsIndexable();
			var res1 = a.L2Norm();
			float res2;
			using (var gpuA = _cuda.CreateVector(a))
				res2 = gpuA.L2Norm();
			Assert.AreEqual(res1, res2);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void VectorMaximumIndex()
		{
			Load();
			var a = _cpu.CreateVector(new[] { 1.0f, 2.0f, 1.0f, 1.0f }).AsIndexable();
			var res1 = a.MaximumIndex();
			int res2;
			using (var gpuA = _cuda.CreateVector(a))
				res2 = gpuA.MaximumIndex();
			Assert.AreEqual(res1, res2);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void VectorMinimumIndex()
		{
			Load();
			var a = _cpu.CreateVector(new[] { 3.0f, -2.0f, 1.0f, 2.0f }).AsIndexable();
			var res1 = a.MinimumIndex();
			int res2;
			using (var gpuA = _cuda.CreateVector(a))
				res2 = gpuA.MinimumIndex();
			Assert.AreEqual(res1, res2);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void VectorAddInPlace()
		{
			Load();
			var a = _cpu.CreateVector(5, i => i * 2).AsIndexable();
			var b = _cpu.CreateVector(5, i => i).AsIndexable();
			IIndexableVector gpuResults;
			using (var gpuA = _cuda.CreateVector(a))
			{
				using var gpuB = _cuda.CreateVector(b);
				gpuA.AddInPlace(gpuB, 2.5f, 3.5f);
				gpuResults = gpuA.AsIndexable();
			}

			a.AddInPlace(b, 2.5f, 3.5f);
			FloatingPointHelper.AssertEqual(a, gpuResults);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void VectorSubtractInPlace()
		{
			Load();
			var a = _cpu.CreateVector(5, i => i * 2).AsIndexable();
			var b = _cpu.CreateVector(5, i => i).AsIndexable();
			IIndexableVector gpuResults;
			using (var gpuA = _cuda.CreateVector(a))
			{
				using var gpuB = _cuda.CreateVector(b);
				gpuA.SubtractInPlace(gpuB, 2.5f, 3.5f);
				gpuResults = gpuA.AsIndexable();
			}

			a.SubtractInPlace(b, 2.5f, 3.5f);
			FloatingPointHelper.AssertEqual(a, gpuResults);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void VectorSqrt()
		{
			Load();
			var a = _cpu.CreateVector(10, i => i * 2).AsIndexable();
			var b = a.Sqrt().AsIndexable();
			IIndexableVector gpuResults;
			using (var gpuA = _cuda.CreateVector(a))
			{
				using var gpuB = gpuA.Sqrt();
				gpuResults = gpuB.AsIndexable();
			}

			FloatingPointHelper.AssertEqual(b, gpuResults);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void VectorGetNewVectorFromIndices()
		{
			Load();
			var a = _cpu.CreateVector(10, i => i * 2).AsIndexable();
			int[] array = { 2, 3, 5 };
			var b = a.GetNewVectorFromIndexes(array).AsIndexable();
			IIndexableVector gpuResults;
			using (var gpuA = _cuda.CreateVector(a))
			{
				using var gpuB = gpuA.GetNewVectorFromIndexes(array);
				gpuResults = gpuB.AsIndexable();
			}

			FloatingPointHelper.AssertEqual(b, gpuResults);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void VectorCopyFrom()
		{
			Load();
			var a = _cpu.CreateVector(10, i => i * 2).AsIndexable();
			var b = _cpu.CreateVector(10, 0).AsIndexable();
			b.CopyFrom(a);
			FloatingPointHelper.AssertEqual(a, b);
			IIndexableVector gpuResults;
			using (var gpuA = _cuda.CreateVector(a))
			{
				using var gpuB = _cuda.CreateVector(10, 0);
				gpuB.CopyFrom(gpuA);
				gpuResults = gpuB.AsIndexable();
			}

			FloatingPointHelper.AssertEqual(a, gpuResults);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void VectorClone()
		{
			Load();
			var a = _cpu.CreateVector(5, i => i).AsIndexable();
			var b = a.Clone().AsIndexable();
			FloatingPointHelper.AssertEqual(a, b);
			IIndexableVector gpuResults;
			using (var gpuA = _cuda.CreateVector(a))
			{
				using var clone = gpuA.Clone();
				gpuResults = clone.AsIndexable();
			}

			FloatingPointHelper.AssertEqual(gpuResults, b);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void VectorMultiply()
		{
			Load();
			var a = _cpu.CreateVector(5, i => i).AsIndexable();
			const float OPERAND = 2f;
			IIndexableVector gpuResults;
			using (var gpuA = _cuda.CreateVector(a))
			{
				gpuA.Multiply(OPERAND);
				gpuResults = gpuA.AsIndexable();
			}

			a.Multiply(OPERAND);
			FloatingPointHelper.AssertEqual(gpuResults, a);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void VectorReadWrite()
		{
			Load();
			var a = _cpu.CreateVector(5, i => i).AsIndexable();

			// test Numerics -> Numerics serialisation
			var serialised = a.Data;
			var b = _cpu.CreateVector(serialised);
			FloatingPointHelper.AssertEqual(a.AsIndexable(), b.AsIndexable());

			// test Numerics -> Cuda serialisation
			using var c = _cuda.CreateVector(serialised);
			FloatingPointHelper.AssertEqual(a.AsIndexable(), c.AsIndexable());

			// test Cuda -> Cuda serialisation
			var serialised2 = c.Data;
			using (var d = _cuda.CreateVector(serialised2))
				FloatingPointHelper.AssertEqual(a.AsIndexable(), d.AsIndexable());

			// test Cuda -> Numerics serialisation
			var e = _cpu.CreateVector(c.Data);
			FloatingPointHelper.AssertEqual(a.AsIndexable(), e.AsIndexable());
			Cleanup();
		}

		[Test, Category("Slow")]
		public void VectorEuclideanDistance()
		{
			Load();
			var distribution = new Normal(0, 5);
			var a = _cpu.CreateVector(5000, i => Convert.ToSingle(distribution.Sample())).AsIndexable();
			var b = _cpu.CreateVector(5000, i => Convert.ToSingle(distribution.Sample())).AsIndexable();
			var distance = a.EuclideanDistance(b);
			float distance2;
			using (var gpuA = _cuda.CreateVector(a))
			{
				using var gpuB = _cuda.CreateVector(b);
				distance2 = gpuA.EuclideanDistance(gpuB);
			}

			Assert.IsTrue(FloatingPointHelper.AlmostEqual2sComplement(distance, distance2, 10));
			Cleanup();
		}

		[Test, Category("Slow")]
		public void VectorCosineDistance()
		{
			Load();
			var rand = new Random(0);
			var a = _cpu.CreateVector(5000, i => Convert.ToSingle(rand.NextDouble())).AsIndexable();
			var b = _cpu.CreateVector(5000, i => Convert.ToSingle(rand.NextDouble())).AsIndexable();
			var distance = a.CosineDistance(b);
			float distance2;
			using (var gpuA = _cuda.CreateVector(a))
			{
				using var gpuB = _cuda.CreateVector(b);
				distance2 = gpuA.CosineDistance(gpuB);
			}

			Assert.IsTrue(FloatingPointHelper.AlmostEqual2sComplement(distance, distance2, 10));
			Cleanup();
		}

		[Test, Category("Slow")]
		public void VectorManhattanDistance()
		{
			Load();
			var distribution = new Normal(0, 5);
			var a = _cpu.CreateVector(5000, i => Convert.ToSingle(distribution.Sample())).AsIndexable();
			var b = _cpu.CreateVector(5000, i => Convert.ToSingle(distribution.Sample())).AsIndexable();
			var distance = a.ManhattanDistance(b);
			float distance2;
			using (var gpuA = _cuda.CreateVector(a))
			{
				using var gpuB = _cuda.CreateVector(b);
				distance2 = gpuA.ManhattanDistance(gpuB);
			}

			Assert.IsTrue(FloatingPointHelper.AlmostEqual2sComplement(distance, distance2, 10));
			Cleanup();
		}

		[Test, Category("Slow")]
		public void VectorMeanSquaredDistance()
		{
			Load();
			var distribution = new Normal(0, 5);
			var a = _cpu.CreateVector(5000, i => Convert.ToSingle(distribution.Sample())).AsIndexable();
			var b = _cpu.CreateVector(5000, i => Convert.ToSingle(distribution.Sample())).AsIndexable();
			var distance = a.MeanSquaredDistance(b);
			float distance2;
			using (var gpuA = _cuda.CreateVector(a))
			{
				using var gpuB = _cuda.CreateVector(b);
				distance2 = gpuA.MeanSquaredDistance(gpuB);
			}

			Assert.IsTrue(FloatingPointHelper.AlmostEqual2sComplement(distance, distance2, 11));
			Cleanup();
		}

		[Test, Category("Slow")]
		public void VectorSquaredEuclideanDistance()
		{
			Load();
			var distribution = new Normal(0, 5);
			var a = _cpu.CreateVector(5000, i => Convert.ToSingle(distribution.Sample())).AsIndexable();
			var b = _cpu.CreateVector(5000, i => Convert.ToSingle(distribution.Sample())).AsIndexable();
			var distance = a.SquaredEuclidean(b);
			float distance2;
			using (var gpuA = _cuda.CreateVector(a))
			{
				using var gpuB = _cuda.CreateVector(b);
				distance2 = gpuA.SquaredEuclidean(gpuB);
			}

			Assert.IsTrue(FloatingPointHelper.AlmostEqual2sComplement(distance, distance2, 11));
			Cleanup();
		}

		[Test, Category("Slow")]
		public void VectorMinMax()
		{
			Load();
			var distribution = new Normal(0, 5);
			var a = _cpu.CreateVector(5000, i => Convert.ToSingle(distribution.Sample())).AsIndexable();
			var minMax = a.GetMinMax();
			(float Min, float Max) minMax2;
			using (var gpuA = _cuda.CreateVector(a))
				minMax2 = gpuA.GetMinMax();
			FloatingPointHelper.AssertEqual(minMax.Min, minMax2.Min);
			FloatingPointHelper.AssertEqual(minMax.Max, minMax2.Max);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void VectorAverage()
		{
			Load();
			var distribution = new Normal(0, 5);
			var a = _cpu.CreateVector(5000, i => Convert.ToSingle(distribution.Sample())).AsIndexable();
			var average = a.Average();
			float average2;
			using (var gpuA = _cuda.CreateVector(a))
				average2 = gpuA.Average();
			FloatingPointHelper.AssertEqual(average, average2, 7);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void VectorL1Norm()
		{
			Load();
			var distribution = new Normal(0, 5);
			var a = _cpu.CreateVector(5000, i => Convert.ToSingle(distribution.Sample())).AsIndexable();
			var v1 = a.L1Norm();
			float v2;
			using (var gpuA = _cuda.CreateVector(a))
				v2 = gpuA.L1Norm();
			FloatingPointHelper.AssertEqual(v1, v2);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void VectorAbs()
		{
			Load();
			var distribution = new Normal(0, 5);
			var a = _cpu.CreateVector(5000, i => Convert.ToSingle(distribution.Sample())).AsIndexable();
			var v1 = a.Abs().AsIndexable();
			IIndexableVector v2;
			using (var gpuA = _cuda.CreateVector(a))
			{
				using var gpuB = gpuA.Abs();
				v2 = gpuB.AsIndexable();
			}

			FloatingPointHelper.AssertEqual(v1, v2);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void VectorLog()
		{
			Load();
			var distribution = new Normal(0, 5);
			var a = _cpu.CreateVector(5000, i => Convert.ToSingle(distribution.Sample())).AsIndexable();
			var v1 = a.Log().AsIndexable();
			IIndexableVector v2;
			using (var gpuA = _cuda.CreateVector(a))
			{
				using var gpuB = gpuA.Log();
				v2 = gpuB.AsIndexable();
			}

			FloatingPointHelper.AssertEqual(v1, v2, 14);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void VectorStdDev()
		{
			Load();
			var distribution = new Normal(0, 5);
			var a = _cpu.CreateVector(5000, i => Convert.ToSingle(distribution.Sample())).AsIndexable();
			var stdDev = a.StdDev(null);
			float stdDev2;
			using (var gpuA = _cuda.CreateVector(a))
				stdDev2 = gpuA.StdDev(null);
			FloatingPointHelper.AssertEqual(stdDev, stdDev2);
			Cleanup();
		}

		void _TestNormalise(NormalisationType type)
		{
			var distribution = new Normal(0, 5);
			IIndexableVector v2;
			var a = _cpu.CreateVector(5000, i => Convert.ToSingle(distribution.Sample()));
			using (var gpuA = _cuda.CreateVector(a.AsIndexable()))
			{
				gpuA.Normalise(type);
				v2 = gpuA.AsIndexable();
			}

			a.Normalise(type);
			var v1 = a.AsIndexable();
			FloatingPointHelper.AssertEqual(v1, v2, 12);
		}

		[Test, Category("Slow")]
		public void VectorFeatureScaleNormalise()
		{
			Load();
			_TestNormalise(NormalisationType.FeatureScale);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void VectorStandardNormalise()
		{
			Load();
			_TestNormalise(NormalisationType.Standard);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void VectorManhattanNormalise()
		{
			Load();
			_TestNormalise(NormalisationType.Manhattan);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void VectorEuclideanNormalise()
		{
			Load();
			_TestNormalise(NormalisationType.Euclidean);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void MultiEuclideanDistance()
		{
			Load();
			var distribution = new Normal(0, 5);
			var a = _cpu.CreateVector(5000, i => Convert.ToSingle(distribution.Sample())).AsIndexable();
			var b = _cpu.CreateVector(5000, i => Convert.ToSingle(distribution.Sample())).AsIndexable();
			var c = _cpu.CreateVector(5000, i => Convert.ToSingle(distribution.Sample())).AsIndexable();
			var distance = a.FindDistances(new[] { b, c }, DistanceMetric.Euclidean).AsIndexable();
			IIndexableVector distance2;
			using (var gpuA = _cuda.CreateVector(a))
			{
				using var gpuB = _cuda.CreateVector(b);
				using var gpuC = _cuda.CreateVector(c);
				using var temp = gpuA.FindDistances(new[] { gpuB, gpuC }, DistanceMetric.Euclidean);
				distance2 = temp.AsIndexable();
			}

			FloatingPointHelper.AssertEqual(distance, distance2, 10);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void MultiManhattanDistance()
		{
			Load();
			var distribution = new Normal(0, 5);
			var a = _cpu.CreateVector(5000, i => Convert.ToSingle(distribution.Sample())).AsIndexable();
			var b = _cpu.CreateVector(5000, i => Convert.ToSingle(distribution.Sample())).AsIndexable();
			var c = _cpu.CreateVector(5000, i => Convert.ToSingle(distribution.Sample())).AsIndexable();
			var distance = a.FindDistances(new[] { b, c }, DistanceMetric.Manhattan).AsIndexable();
			IIndexableVector distance2;
			using (var gpuA = _cuda.CreateVector(a))
			{
				using var gpuB = _cuda.CreateVector(b);
				using var gpuC = _cuda.CreateVector(c);
				using var temp = gpuA.FindDistances(new[] { gpuB, gpuC }, DistanceMetric.Manhattan);
				distance2 = temp.AsIndexable();
			}

			FloatingPointHelper.AssertEqual(distance, distance2, 18);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void MultiCosineDistance()
		{
			Load();
			var distribution = new Normal(0, 5);
			var a = _cpu.CreateVector(5000, i => Convert.ToSingle(distribution.Sample())).AsIndexable();
			var b = _cpu.CreateVector(5000, i => Convert.ToSingle(distribution.Sample())).AsIndexable();
			var c = _cpu.CreateVector(5000, i => Convert.ToSingle(distribution.Sample())).AsIndexable();
			var distance = a.FindDistances(new[] { b, c }, DistanceMetric.Cosine).AsIndexable();
			IIndexableVector distance2;
			using (var gpuA = _cuda.CreateVector(a))
			{
				using var gpuB = _cuda.CreateVector(b);
				using var gpuC = _cuda.CreateVector(c);
				using var temp = gpuA.FindDistances(new[] { gpuB, gpuC }, DistanceMetric.Cosine);
				distance2 = temp.AsIndexable();
			}

			FloatingPointHelper.AssertEqual(distance, distance2, 10);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void MultiCosineDistance2()
		{
			Load();
			var distribution = new Normal(0, 5);
			float[] dataNorm1 = null;
			float[] dataNorm2 = null;
			var a = _cpu.CreateVector(5000, i => Convert.ToSingle(distribution.Sample())).AsIndexable();
			var b = _cpu.CreateVector(5000, i => Convert.ToSingle(distribution.Sample())).AsIndexable();
			var c = _cpu.CreateVector(5000, i => Convert.ToSingle(distribution.Sample())).AsIndexable();
			var distance = a.CosineDistance(new[] { b, c }, ref dataNorm1).AsIndexable();
			IIndexableVector distance2;
			using (var gpuA = _cuda.CreateVector(a))
			{
				using var gpuB = _cuda.CreateVector(b);
				using var gpuC = _cuda.CreateVector(c);
				using var temp = gpuA.CosineDistance(new[] { gpuB, gpuC }, ref dataNorm2);
				distance2 = temp.AsIndexable();
			}

			FloatingPointHelper.AssertEqual(distance, distance2, 10);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void VectorAddScalar()
		{
			Load();
			var a = _cpu.CreateVector(1000, i => i).AsIndexable();
			IIndexableVector gpuResults;
			using (var gpuA = _cuda.CreateVector(a))
			{
				gpuA.Add(0.5f);
				gpuResults = gpuA.AsIndexable();
			}

			a.Add(0.5f);
			FloatingPointHelper.AssertEqual(a, gpuResults);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void VectorSigmoid()
		{
			Load();
			var a = _cpu.CreateVector(1000, i => i).AsIndexable();
			var results = a.Sigmoid().AsIndexable();
			IIndexableVector gpuResults;
			using (var gpuA = _cuda.CreateVector(a))
			{
				using var gpuB = gpuA.Sigmoid();
				gpuResults = gpuB.AsIndexable();
			}

			FloatingPointHelper.AssertEqual(results, gpuResults);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void MatrixVectorMultiply()
		{
			Load();
			var a = _cpu.CreateMatrix(256, 256, (x, y) => x * y).AsIndexable();
			var b = _cpu.CreateVector(256, i => i * 0.5f).AsIndexable();
			var c = a.Multiply(b).AsIndexable();
			IIndexableMatrix gpuResults;
			using (var gpuA = _cuda.CreateMatrix(a))
			{
				using var gpuB = _cuda.CreateVector(b);
				using var gpuC = gpuA.Multiply(gpuB);
				gpuResults = gpuC.AsIndexable();
			}

			FloatingPointHelper.AssertEqual(c, gpuResults);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void VectorSplit()
		{
			Load();
			const int BLOCK_COUNT = 3;
			var a = _cpu.CreateVector(12, i => i).AsIndexable();
			var cpuResult = a.Split(BLOCK_COUNT).Select(v => v.AsIndexable()).ToList();
			var gpuResult = new List<IIndexableVector>();
			using (var gpuA = _cuda.CreateVector(a))
			{
				var split = gpuA.Split(BLOCK_COUNT);
				foreach (var item in split)
				{
					gpuResult.Add(item.AsIndexable());
					item.Dispose();
				}
			}

			for (var i = 0; i < cpuResult.Count; i++)
			{
				Assert.IsTrue(cpuResult[i].Count == 4);
				FloatingPointHelper.AssertEqual(cpuResult[i], gpuResult[i]);
			}

			Cleanup();
		}

		[Test, Category("Slow")]
		public void VectorSoftMax()
		{
			Load();
			var distribution = new Normal(0, 5);
			var a = _cpu.CreateVector(128, i => (float)distribution.Sample()).AsIndexable();
			var cpuResult = a.Softmax().AsIndexable();
			IIndexableVector result;
			using (var gpuA = _cuda.CreateVector(a))
			{
				var gpuResult = gpuA.Softmax();
				result = gpuResult.AsIndexable();
			}

			FloatingPointHelper.AssertEqual(result, cpuResult);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void VectorSoftMaxDerivative()
		{
			Load();
			var distribution = new Normal(0, 5);
			var a = _cpu.CreateVector(128, i => (float)distribution.Sample()).AsIndexable();
			var cpuResult = a.SoftmaxDerivative().AsIndexable();
			IIndexableMatrix result;
			using (var gpuA = _cuda.CreateVector(a))
			{
				var gpuResult = gpuA.SoftmaxDerivative();
				result = gpuResult.AsIndexable();
			}

			FloatingPointHelper.AssertEqual(result, cpuResult);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void VectorReverse()
		{
			Load();
			var distribution = new Normal(0, 5);
			var a = _cpu.CreateVector(128, i => (float)distribution.Sample()).AsIndexable();
			var cpuResult = a.Reverse().AsIndexable();
			IIndexableVector result;
			using (var gpuA = _cuda.CreateVector(a))
			{
				var gpuResult = gpuA.Reverse();
				result = gpuResult.AsIndexable();
			}

			FloatingPointHelper.AssertEqual(result, cpuResult);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void VectorRotate()
		{
			Load();
			var a = _cpu.CreateVector(4, i => i + 1).AsIndexable();
			using var gpuA = _cuda.CreateVector(a);
			a.RotateInPlace();
			gpuA.RotateInPlace();
			FloatingPointHelper.AssertEqual(gpuA.AsIndexable(), a.AsIndexable());
			Cleanup();
		}

		[Test, Category("Slow")]
		public void VectorRotate2()
		{
			Load();
			const int blockCount = 2;
			var a = _cpu.CreateVector(8, i => i + 1).AsIndexable();
			using var gpuA = _cuda.CreateVector(a);
			a.RotateInPlace(blockCount);
			gpuA.RotateInPlace(blockCount);
			FloatingPointHelper.AssertEqual(gpuA.AsIndexable(), a.AsIndexable());
			Cleanup();
		}

		[Test, Category("Slow")]
		public void TestFinite()
		{
			Load();
			var vector = _cpu.CreateVector(new[] { 0f, 1f, 2f, 3f, -1f });
			Assert.IsTrue(vector.IsEntirelyFinite());
			using var gpuVector = _cuda.CreateVector(vector.AsIndexable());
			Assert.IsTrue(gpuVector.IsEntirelyFinite());
			Cleanup();
		}

		[Test, Category("Slow")]
		public void TestFinite2()
		{
			Load();
			var vector = _cpu.CreateVector(new[] { 0f, 1f, 2f, 3f, -1f, float.Epsilon });
			Assert.IsTrue(vector.IsEntirelyFinite());
			using var gpuVector = _cuda.CreateVector(vector.AsIndexable());
			Assert.IsTrue(gpuVector.IsEntirelyFinite());
			Cleanup();
		}

		[Test, Category("Slow")]
		public void TestNotFinite()
		{
			Load();
			var vector = _cpu.CreateVector(new[] { 0f, 1f, 2f, 3f, float.NaN });
			Assert.IsFalse(vector.IsEntirelyFinite());
			using var gpuVector = _cuda.CreateVector(vector.AsIndexable());
			Assert.IsFalse(gpuVector.IsEntirelyFinite());
			Cleanup();
		}

		[Test, Category("Slow")]
		public void TestNotFinite2()
		{
			Load();
			var vector = _cpu.CreateVector(new[] { 0f, 1f, 2f, 3f, float.NegativeInfinity });
			Assert.IsFalse(vector.IsEntirelyFinite());
			using var gpuVector = _cuda.CreateVector(vector.AsIndexable());
			Assert.IsFalse(gpuVector.IsEntirelyFinite());
			Cleanup();
		}

		[Test, Category("Slow")]
		public void TestNotFinite3()
		{
			Load();
			var vector = _cpu.CreateVector(new[] { 0f, 1f, 2f, 3f, float.PositiveInfinity });
			Assert.IsFalse(vector.IsEntirelyFinite());
			using var gpuVector = _cuda.CreateVector(vector.AsIndexable());
			Assert.IsFalse(gpuVector.IsEntirelyFinite());
			Cleanup();
		}
	}
}