using System;
using System.Linq;
using BrightWire.Cuda;
using BrightWire.Tests.Helper;
using MathNet.Numerics.Distributions;
using NUnit.Framework;

namespace BrightWire.Tests
{
	public class CudaMatrixTests
	{
		static ILinearAlgebraProvider _cuda;
		static ILinearAlgebraProvider _cpu;

		public static void Load()
		{
			_cuda = BrightWireGpuProvider.CreateLinearAlgebra(false);
			_cpu = BrightWireProvider.CreateLinearAlgebra(false);
		}

		public void _MatrixMultiplication(int rowsA, int columnsArowsB, int columnsB)
		{
			var rand = new Random(1);
			var a = _cpu.
				CreateMatrix(rowsA, columnsArowsB, (j, k) => Convert.ToSingle(rand.NextDouble())).
				AsIndexable();
			var b = _cpu.
				CreateMatrix(columnsArowsB, columnsB, (j, k) => Convert.ToSingle(rand.NextDouble())).
				AsIndexable();
			var cpuResults = a.Multiply(b);
			IIndexableMatrix gpuResults;
			using (var gpuA = _cuda.CreateMatrix(a))
			{
				using var gpuB = _cuda.CreateMatrix(b);
				using var gpuC = gpuA.Multiply(gpuB);
				gpuResults = gpuC.AsIndexable();
			}

			FloatingPointHelper.AssertEqual(gpuResults, cpuResults.AsIndexable());
			_cuda.Dispose();
		}

		[Test, Category("Slow")]
		public void TestMatrixCreationFromRows()
		{
			Load();
			var values = new[]
			{
				Enumerable.Range(0, 10).Select(v => (float)v).ToArray(),
				Enumerable.Range(0, 10).Select(v => (float)v * 2).ToArray(),
				Enumerable.Range(0, 10).Select(v => (float)v * 3).ToArray()
			};
			var cpuRowList = values.Select(v => _cpu.CreateVector(v)).ToList();
			var cpuMatrix = _cpu.CreateMatrixFromRows(cpuRowList);
			var gpuRowList = values.Select(v => _cuda.CreateVector(v)).ToList();
			using (var gpuMatrix = _cuda.CreateMatrixFromRows(gpuRowList))
			{
				FloatingPointHelper.AssertEqual(cpuMatrix.AsIndexable(), gpuMatrix.AsIndexable());
			}

			gpuRowList.ForEach(v => v.Dispose());
			Cleanup();
		}

		[Test, Category("Slow")]
		public void TestMatrixCreationFromColumns()
		{
			Load();
			var values = new[]
			{
				Enumerable.Range(0, 10).Select(v => (float)v).ToArray(),
				Enumerable.Range(0, 10).Select(v => (float)v * 2).ToArray(),
				Enumerable.Range(0, 10).Select(v => (float)v * 3).ToArray()
			};
			var cpuRowList = values.Select(v => _cpu.CreateVector(v)).ToList();
			var cpuMatrix = _cpu.CreateMatrixFromColumns(cpuRowList);
			var gpuRowList = values.Select(v => _cuda.CreateVector(v)).ToList();
			using (var gpuMatrix = _cuda.CreateMatrixFromColumns(gpuRowList))
			{
				FloatingPointHelper.AssertEqual(cpuMatrix.AsIndexable(), gpuMatrix.AsIndexable());
			}

			gpuRowList.ForEach(v => v.Dispose());
			Cleanup();
		}

		[Test, Category("Slow")]
		public void MatrixMultiplication()
		{
			Load();
			_MatrixMultiplication(5, 2, 5);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void MatrixMultiplication2()
		{
			Load();
			_MatrixMultiplication(500, 200, 500);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void MatrixMultiplication3()
		{
			Load();
			_MatrixMultiplication(5, 5, 5);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void MatrixMultiplication4()
		{
			Load();
			_MatrixMultiplication(5, 10, 5);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void MatrixMultiplication5()
		{
			Load();
			_MatrixMultiplication(5, 10, 2);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void MatrixMultiplication6()
		{
			Load();
			_MatrixMultiplication(50, 10, 2);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void MatrixMultiplication7()
		{
			Load();
			_MatrixMultiplication(2, 10, 20);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void MatrixMultiplication8()
		{
			Load();
			_MatrixMultiplication(20, 1, 19);
			Cleanup();
		}

		public void _Transpose(int rows, int columns)
		{
			var a = _cpu.CreateMatrix(rows, columns, (j, k) => k).AsIndexable();
			var aT = a.Transpose();
			IIndexableMatrix gpuResults;
			using (var gpuA = _cuda.CreateMatrix(a))
			{
				using var gpuAT = gpuA.Transpose();
				gpuResults = gpuAT.AsIndexable();
			}

			FloatingPointHelper.AssertEqual(gpuResults, aT.AsIndexable());
		}

		[Test, Category("Slow")]
		public void MatrixTranspose()
		{
			Load();
			_Transpose(2, 5);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void MatrixTranspose2()
		{
			Load();
			_Transpose(5, 2);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void MatrixTranspose3()
		{
			Load();
			_Transpose(500, 2);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void MatrixTranspose4()
		{
			Load();
			_Transpose(2, 500);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void MatrixTranspose5()
		{
			Load();
			_Transpose(20, 20);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void MatrixTranspose6()
		{
			Load();
			_Transpose(500, 500);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void MatrixTransposeAndMultiplication()
		{
			Load();
			var a = _cpu.CreateMatrix(2, 5, (j, k) => k).AsIndexable();
			var b = _cpu.CreateMatrix(3, 5, (j, k) => j).AsIndexable();
			var cpuResults = a.TransposeAndMultiply(b);
			IIndexableMatrix gpuResults;
			using (var gpuA = _cuda.CreateMatrix(a))
			{
				using var gpuB = _cuda.CreateMatrix(b);
				using var gpuC = gpuA.TransposeAndMultiply(gpuB);
				gpuResults = gpuC.AsIndexable();
			}

			FloatingPointHelper.AssertEqual(gpuResults, cpuResults.AsIndexable());
			Cleanup();
		}

		[Test, Category("Slow")]
		public void MatrixTransposeAndMultiplication2()
		{
			Load();
			var a = _cpu.CreateMatrix(2, 6, (j, k) => k).AsIndexable();
			var b = _cpu.CreateMatrix(2, 5, (j, k) => j).AsIndexable();
			var cpuResults = a.TransposeThisAndMultiply(b);
			IIndexableMatrix gpuResults;
			using (var gpuA = _cuda.CreateMatrix(a))
			{
				using var gpuB = _cuda.CreateMatrix(b);
				using var gpuC = gpuA.TransposeThisAndMultiply(gpuB);
				gpuResults = gpuC.AsIndexable();
			}

			FloatingPointHelper.AssertEqual(gpuResults, cpuResults.AsIndexable());
			Cleanup();
		}

		[Test, Category("Slow")]
		public void MatrixAdd()
		{
			Load();
			var a = _cpu.CreateMatrix(2, 5, (j, k) => k).AsIndexable();
			var b = _cpu.CreateMatrix(2, 5, (j, k) => j).AsIndexable();
			var cpuResults = a.Add(b);
			IIndexableMatrix gpuResults;
			using (var gpuA = _cuda.CreateMatrix(a))
			{
				using var gpuB = _cuda.CreateMatrix(b);
				using var gpuC = gpuA.Add(gpuB);
				gpuResults = gpuC.AsIndexable();
			}

			FloatingPointHelper.AssertEqual(gpuResults, cpuResults.AsIndexable());
			Cleanup();
		}

		[Test, Category("Slow")]
		public void MatrixSubtract()
		{
			Load();
			var a = _cpu.CreateMatrix(2, 5, (j, k) => k).AsIndexable();
			var b = _cpu.CreateMatrix(2, 5, (j, k) => j).AsIndexable();
			IIndexableMatrix gpuResults, gpuResults2;
			using (var gpuA = _cuda.CreateMatrix(a))
			{
				using var gpuB = _cuda.CreateMatrix(b);
				var aStr = gpuA.ToString();
				var bStr = gpuB.ToString();
				using (var gpuC = gpuA.Subtract(gpuB))
				{
					using var gpuD = gpuB.Subtract(gpuA);
					gpuResults = gpuC.AsIndexable();
					gpuResults2 = gpuD.AsIndexable();
				}

				Assert.AreEqual(aStr, gpuA.ToString());
				Assert.AreEqual(bStr, gpuB.ToString());
			}

			var cpuResults = a.Subtract(b);
			var cpuResults2 = b.Subtract(a);
			FloatingPointHelper.AssertEqual(gpuResults, cpuResults.AsIndexable());
			FloatingPointHelper.AssertEqual(gpuResults2, cpuResults2.AsIndexable());
			Cleanup();
		}

		[Test, Category("Slow")]
		public void MatrixPointwiseMultiply()
		{
			Load();
			var a = _cpu.CreateMatrix(2, 5, (j, k) => k).AsIndexable();
			var b = _cpu.CreateMatrix(2, 5, (j, k) => j).AsIndexable();
			var cpuResults = a.PointwiseMultiply(b);
			IIndexableMatrix gpuResults;
			using (var gpuA = _cuda.CreateMatrix(a))
			{
				using var gpuB = _cuda.CreateMatrix(b);
				using var gpuC = gpuA.PointwiseMultiply(gpuB);
				gpuResults = gpuC.AsIndexable();
			}

			FloatingPointHelper.AssertEqual(gpuResults, cpuResults.AsIndexable());
			Cleanup();
		}

		[Test, Category("Slow")]
		public void MatrixPointwiseDivide()
		{
			Load();
			var a = _cpu.CreateMatrix(2, 5, (j, k) => k + 1).AsIndexable();
			var b = _cpu.CreateMatrix(2, 5, (j, k) => j + 1).AsIndexable();
			var cpuResults = a.PointwiseDivide(b);
			IIndexableMatrix gpuResults;
			using (var gpuA = _cuda.CreateMatrix(a))
			{
				using var gpuB = _cuda.CreateMatrix(b);
				using var gpuC = gpuA.PointwiseDivide(gpuB);
				gpuResults = gpuC.AsIndexable();
			}

			FloatingPointHelper.AssertEqual(gpuResults, cpuResults.AsIndexable());
			Cleanup();
		}

		[Test, Category("Slow")]
		public void MatrixSqrt()
		{
			Load();
			var a = _cpu.CreateMatrix(2, 5, (j, k) => k + 1).AsIndexable();
			const float adjustment = 1e-8f;
			a[0, 0] = -adjustment;
			var cpuResults = a.Sqrt();
			IIndexableMatrix gpuResults;
			using (var gpuA = _cuda.CreateMatrix(a))
			{
				using var gpuC = gpuA.Sqrt();
				gpuResults = gpuC.AsIndexable();
			}

			FloatingPointHelper.AssertEqual(gpuResults, cpuResults.AsIndexable());
			Cleanup();
		}

		[Test, Category("Slow")]
		public void MatrixMultiplyScalar()
		{
			Load();
			var a = _cpu.CreateMatrix(2, 5, (j, k) => k).AsIndexable();
			const float scalar = 2.5f;
			IIndexableMatrix gpuResults;
			using (var gpuA = _cuda.CreateMatrix(a))
			{
				gpuA.Multiply(scalar);
				gpuResults = gpuA.AsIndexable();
			}

			a.Multiply(scalar);
			FloatingPointHelper.AssertEqual(gpuResults, a);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void MatrixColumn()
		{
			Load();
			const int INDEX = 7;
			var a = _cpu.CreateMatrix(13, 17, (j, k) => (j + 1) * (k + 1)).AsIndexable();
			var row = a.Column(INDEX).AsIndexable();
			IIndexableVector gpuResults;
			using (var gpuA = _cuda.CreateMatrix(a))
			{
				using var gpuCol = gpuA.Column(INDEX);
				gpuResults = gpuCol.AsIndexable();
			}

			FloatingPointHelper.AssertEqual(gpuResults, row);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void MatrixRow()
		{
			Load();
			const int INDEX = 11;
			var a = _cpu.CreateMatrix(20, 50, (j, k) => k * j).AsIndexable();
			var row = a.Row(INDEX).AsIndexable();
			IIndexableVector gpuResults;
			using (var gpuA = _cuda.CreateMatrix(a))
			{
				using var gpuRow = gpuA.Row(INDEX);
				gpuResults = gpuRow.AsIndexable();
			}

			FloatingPointHelper.AssertEqual(gpuResults, row);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void MatrixRowSums()
		{
			Load();
			var a = _cpu.CreateMatrix(2, 5, (j, k) => k).AsIndexable();
			var rowSums = a.RowSums().AsIndexable();
			IIndexableVector gpuResults;
			using (var gpuA = _cuda.CreateMatrix(a))
			{
				using var gpuRowSums = gpuA.RowSums();
				gpuResults = gpuRowSums.AsIndexable();
			}

			FloatingPointHelper.AssertEqual(gpuResults, rowSums);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void MatrixColumnSums()
		{
			Load();
			var a = _cpu.CreateMatrix(2, 5, (j, k) => k).AsIndexable();
			var colSums = a.ColumnSums().AsIndexable();
			IIndexableVector gpuResults;
			using (var gpuA = _cuda.CreateMatrix(a))
			{
				using var gpuColSums = gpuA.ColumnSums();
				gpuResults = gpuColSums.AsIndexable();
			}

			FloatingPointHelper.AssertEqual(gpuResults, colSums);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void MatrixAddInPlace()
		{
			Load();
			var a = _cpu.CreateMatrix(2, 5, (j, k) => k).AsIndexable();
			var b = _cpu.CreateMatrix(2, 5, (j, k) => j).AsIndexable();
			IIndexableMatrix gpuResults;
			using (var gpuA = _cuda.CreateMatrix(a))
			{
				using var gpuB = _cuda.CreateMatrix(b);
				gpuA.AddInPlace(gpuB, 1.5f, 2.5f);
				gpuResults = gpuA.AsIndexable();
			}

			a.AddInPlace(b, 1.5f, 2.5f);
			FloatingPointHelper.AssertEqual(gpuResults, a);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void MatrixSubtractInPlace()
		{
			Load();
			var a = _cpu.CreateMatrix(2, 5, (j, k) => k).AsIndexable();
			var b = _cpu.CreateMatrix(2, 5, (j, k) => j).AsIndexable();
			IIndexableMatrix gpuResults;
			using (var gpuA = _cuda.CreateMatrix(a))
			{
				using var gpuB = _cuda.CreateMatrix(b);
				gpuA.SubtractInPlace(gpuB, 1.5f, 2.5f);
				gpuResults = gpuA.AsIndexable();
			}

			a.SubtractInPlace(b, 1.5f, 2.5f);
			FloatingPointHelper.AssertEqual(gpuResults, a);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void MatrixAddToEachRow()
		{
			Load();
			var a = _cpu.CreateMatrix(2, 5, (j, k) => k * j).AsIndexable();
			var b = _cpu.CreateVector(5, i => i).AsIndexable();
			IIndexableMatrix gpuResults;
			using (var gpuA = _cuda.CreateMatrix(a))
			{
				using var gpuB = _cuda.CreateVector(b);
				gpuA.AddToEachRow(gpuB);
				gpuResults = gpuA.AsIndexable();
			}

			a.AddToEachRow(b);
			FloatingPointHelper.AssertEqual(gpuResults, a);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void MatrixAddToEachColumn()
		{
			Load();
			var a = _cpu.CreateMatrix(2, 5, (j, k) => k * j).AsIndexable();
			var b = _cpu.CreateVector(2, i => i + 5).AsIndexable();
			IIndexableMatrix gpuResults;
			using (var gpuA = _cuda.CreateMatrix(a))
			{
				using var gpuB = _cuda.CreateVector(b);
				gpuA.AddToEachColumn(gpuB);
				gpuResults = gpuA.AsIndexable();
			}

			a.AddToEachColumn(b);
			FloatingPointHelper.AssertEqual(gpuResults, a);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void MatrixSigmoidActivation()
		{
			Load();
			var normalDistribution = new Normal(0, 1);
			var a = _cpu.CreateMatrix(3, 7, (j, k) => Convert.ToSingle(normalDistribution.Sample())).
				AsIndexable();
			var b = a.SigmoidActivation().AsIndexable();
			IIndexableMatrix gpuResults;
			using (var gpuA = _cuda.CreateMatrix(a))
			{
				using var sigmoid = gpuA.SigmoidActivation();
				gpuResults = sigmoid.AsIndexable();
			}

			FloatingPointHelper.AssertEqual(gpuResults, b);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void MatrixSigmoidDerivative()
		{
			Load();
			var normalDistribution = new Normal(0, 1);
			var a = _cpu.CreateMatrix(3, 7, (j, k) => Convert.ToSingle(normalDistribution.Sample())).
				AsIndexable();
			var b = a.SigmoidDerivative().AsIndexable();
			IIndexableMatrix gpuResults;
			using (var gpuA = _cuda.CreateMatrix(a))
			{
				using var sigmoid = gpuA.SigmoidDerivative();
				gpuResults = sigmoid.AsIndexable();
			}

			FloatingPointHelper.AssertEqual(gpuResults, b);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void MatrixTanhActivation()
		{
			Load();
			var normalDistribution = new Normal(0, 1);
			var a = _cpu.CreateMatrix(3, 7, (j, k) => Convert.ToSingle(normalDistribution.Sample())).
				AsIndexable();
			var b = a.TanhActivation().AsIndexable();
			IIndexableMatrix gpuResults;
			using (var gpuA = _cuda.CreateMatrix(a))
			{
				using var tanh = gpuA.TanhActivation();
				gpuResults = tanh.AsIndexable();
			}

			FloatingPointHelper.AssertEqual(gpuResults, b);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void MatrixTanhDerivative()
		{
			Load();
			var normalDistribution = new Normal(0, 1);
			var a = _cpu.CreateMatrix(3, 7, (j, k) => Convert.ToSingle(normalDistribution.Sample())).
				AsIndexable();
			var b = a.TanhDerivative().AsIndexable();
			IIndexableMatrix gpuResults;
			using (var gpuA = _cuda.CreateMatrix(a))
			{
				using var tanh = gpuA.TanhDerivative();
				gpuResults = tanh.AsIndexable();
			}

			FloatingPointHelper.AssertEqual(gpuResults, b);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void MatrixRELUActivation()
		{
			Load();
			var normalDistribution = new Normal(0, 1);
			var a = _cpu.CreateMatrix(3, 7, (j, k) => Convert.ToSingle(normalDistribution.Sample())).
				AsIndexable();
			var b = a.ReluActivation().AsIndexable();
			IIndexableMatrix gpuResults;
			using (var gpuA = _cuda.CreateMatrix(a))
			{
				using var relu = gpuA.ReluActivation();
				gpuResults = relu.AsIndexable();
			}

			FloatingPointHelper.AssertEqual(gpuResults, b);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void MatrixRELUDerivative()
		{
			Load();
			var normalDistribution = new Normal(0, 1);
			var a = _cpu.CreateMatrix(3, 7, (j, k) => Convert.ToSingle(normalDistribution.Sample())).
				AsIndexable();
			var b = a.ReluDerivative().AsIndexable();
			IIndexableMatrix gpuResults;
			using (var gpuA = _cuda.CreateMatrix(a))
			{
				using var relu = gpuA.ReluDerivative();
				gpuResults = relu.AsIndexable();
			}

			FloatingPointHelper.AssertEqual(gpuResults, b);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void MatrixLeakyRELUActivation()
		{
			Load();
			var normalDistribution = new Normal(0, 1);
			var a = _cpu.CreateMatrix(3, 7, (j, k) => Convert.ToSingle(normalDistribution.Sample())).
				AsIndexable();
			var b = a.LeakyReluActivation().AsIndexable();
			IIndexableMatrix gpuResults;
			using (var gpuA = _cuda.CreateMatrix(a))
			{
				using var relu = gpuA.LeakyReluActivation();
				gpuResults = relu.AsIndexable();
			}

			FloatingPointHelper.AssertEqual(gpuResults, b);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void MatrixLeakyRELUDerivative()
		{
			Load();
			var normalDistribution = new Normal(0, 1);
			var a = _cpu.CreateMatrix(3, 7, (j, k) => Convert.ToSingle(normalDistribution.Sample())).
				AsIndexable();
			var b = a.LeakyReluDerivative().AsIndexable();
			IIndexableMatrix gpuResults;
			using (var gpuA = _cuda.CreateMatrix(a))
			{
				using var relu = gpuA.LeakyReluDerivative();
				gpuResults = relu.AsIndexable();
			}

			FloatingPointHelper.AssertEqual(gpuResults, b);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void MatrixSoftmaxActivation()
		{
			Load();
			var normalDistribution = new Normal(0, 1);
			var a = _cpu.CreateMatrix(3, 7, (j, k) => Convert.ToSingle(normalDistribution.Sample())).
				AsIndexable();
			var b = a.SoftmaxActivation().AsIndexable();
			IIndexableMatrix gpuResults;
			using (var gpuA = _cuda.CreateMatrix(a))
			{
				using var softmax = gpuA.SoftmaxActivation();
				gpuResults = softmax.AsIndexable();
			}

			FloatingPointHelper.AssertEqual(gpuResults, b);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void MatrixNewMatrixFromRows()
		{
			Load();
			var rows = new[] { 7, 8, 9 };
			var a = _cpu.CreateMatrix(13, 17, (j, k) => (k + 1) * (j + 1)).AsIndexable();
			var results = a.GetNewMatrixFromRows(rows).AsIndexable();
			IIndexableMatrix gpuResults;
			using (var gpuA = _cuda.CreateMatrix(a))
			{
				using var m = gpuA.GetNewMatrixFromRows(rows);
				gpuResults = m.AsIndexable();
			}

			FloatingPointHelper.AssertEqual(gpuResults, results);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void MatrixNewMatrixFromColumns()
		{
			Load();
			var cols = new[] { 1, 2, 9 };
			var a = _cpu.CreateMatrix(12, 13, (j, k) => (k + 1) * (j + 1)).AsIndexable();
			var results = a.GetNewMatrixFromColumns(cols).AsIndexable();
			IIndexableMatrix gpuResults;
			using (var gpuA = _cuda.CreateMatrix(a))
			{
				using var m = gpuA.GetNewMatrixFromColumns(cols);
				gpuResults = m.AsIndexable();
			}

			FloatingPointHelper.AssertEqual(gpuResults, results);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void MatrixClearRows()
		{
			Load();
			var rows = new[] { 1, 2, 9 };
			var a = _cpu.CreateMatrix(13, 12, (j, k) => k + 1).AsIndexable();
			a.ClearRows(rows);
			IIndexableMatrix gpuResults;
			using (var gpuA = _cuda.CreateMatrix(a))
			{
				a.ClearRows(rows);
				gpuResults = a.AsIndexable();
			}

			FloatingPointHelper.AssertEqual(gpuResults, a);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void MatrixClearColumns()
		{
			Load();
			var cols = new[] { 1, 2, 7 };
			var a = _cpu.CreateMatrix(18, 13, (j, k) => k + 1).AsIndexable();
			a.ClearColumns(cols);
			IIndexableMatrix gpuResults;
			using (var gpuA = _cuda.CreateMatrix(a))
			{
				gpuA.ClearColumns(cols);
				gpuResults = gpuA.AsIndexable();
			}

			FloatingPointHelper.AssertEqual(gpuResults, a);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void MatrixClear()
		{
			Load();
			var a = _cpu.CreateMatrix(15, 23, (j, k) => k + 1).AsIndexable();
			IIndexableMatrix gpuResults;
			using (var gpuA = _cuda.CreateMatrix(a))
			{
				gpuA.Clear();
				gpuResults = gpuA.AsIndexable();
			}

			a.Clear();
			FloatingPointHelper.AssertEqual(gpuResults, a);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void MatrixClone()
		{
			Load();
			var a = _cpu.CreateMatrix(12, 7, (j, k) => k + 1).AsIndexable();
			var b = a.Clone().AsIndexable();
			FloatingPointHelper.AssertEqual(a, b);
			IIndexableMatrix gpuResults;
			using (var gpuA = _cuda.CreateMatrix(a))
			{
				using var clone = gpuA.Clone();
				gpuResults = clone.AsIndexable();
			}

			FloatingPointHelper.AssertEqual(gpuResults, b);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void MatrixReadWrite()
		{
			Load();
			var a = _cpu.CreateMatrix(7, 20, (x, y) => x * 10 + y).AsIndexable();

			// test Numerics -> Numerics serialisation
			var serialised = a.Data;
			var b = _cpu.CreateMatrix(serialised);
			FloatingPointHelper.AssertEqual(a.AsIndexable(), b.AsIndexable());

			// test Numerics -> Cuda serialisation
			using var c = _cuda.CreateMatrix(serialised);
			FloatingPointHelper.AssertEqual(a.AsIndexable(), c.AsIndexable());

			// test Cuda -> Cuda serialisation
			var serialised2 = c.Data;
			using (var d = _cuda.CreateMatrix(serialised2))
				FloatingPointHelper.AssertEqual(a.AsIndexable(), d.AsIndexable());

			// test Cuda -> Numerics serialisation
			var e = _cpu.CreateMatrix(c.Data);
			FloatingPointHelper.AssertEqual(a.AsIndexable(), e.AsIndexable());
			Cleanup();
		}

		[Test, Category("Slow")]
		public void MatrixConcatColumns()
		{
			Load();
			var rand = new Random();
			var a = _cpu.CreateMatrix(4000, 300, (x, y) => Convert.ToSingle(rand.NextDouble())).
				AsIndexable();
			var b = _cpu.CreateMatrix(200, 300, (x, y) => Convert.ToSingle(rand.NextDouble())).
				AsIndexable();
			var c = a.ConcatColumns(b).AsIndexable();
			IIndexableMatrix gpuResults;
			using (var gpuA = _cuda.CreateMatrix(a))
			{
				using var gpuB = _cuda.CreateMatrix(b);
				using var concat = gpuA.ConcatColumns(gpuB);
				gpuResults = concat.AsIndexable();
			}

			FloatingPointHelper.AssertEqual(c, gpuResults);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void MatrixConcatRows()
		{
			Load();
			var rand = new Random();
			var a = _cpu.CreateMatrix(300, 4000, (x, y) => Convert.ToSingle(rand.NextDouble())).
				AsIndexable();
			var b = _cpu.CreateMatrix(300, 200, (x, y) => Convert.ToSingle(rand.NextDouble())).
				AsIndexable();
			var c = a.ConcatRows(b).AsIndexable();
			IIndexableMatrix gpuResults;
			using (var gpuA = _cuda.CreateMatrix(a))
			{
				using var gpuB = _cuda.CreateMatrix(b);
				using var concat = gpuA.ConcatRows(gpuB);
				gpuResults = concat.AsIndexable();
			}

			FloatingPointHelper.AssertEqual(c, gpuResults);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void MatrixSplitColumns()
		{
			Load();
			const int POSITION = 2000;
			var rand = new Random();
			var a = _cpu.CreateMatrix(6000, 3000, (x, y) => Convert.ToSingle(rand.NextDouble())).
				AsIndexable();
			var r = a.SplitAtRow(POSITION);
			IIndexableMatrix gpuResults1, gpuResults2;
			using (var gpuA = _cuda.CreateMatrix(a))
			{
				var r2 = gpuA.SplitAtRow(POSITION);
				using var m1 = r2.Top;
				using var m2 = r2.Bottom;
				gpuResults1 = m1.AsIndexable();
				gpuResults2 = m2.AsIndexable();
			}

			FloatingPointHelper.AssertEqual(gpuResults1, r.Top.AsIndexable());
			FloatingPointHelper.AssertEqual(gpuResults2, r.Bottom.AsIndexable());
			Cleanup();
		}

		[Test, Category("Slow")]
		public void MatrixSplitRows()
		{
			Load();
			const int POSITION = 2000;
			var rand = new Random();
			var a = _cpu.CreateMatrix(6000, 3000, (x, y) => Convert.ToSingle(rand.NextDouble())).
				AsIndexable();
			var r = a.SplitAtColumn(POSITION);
			IIndexableMatrix gpuResults1, gpuResults2;
			using (var gpuA = _cuda.CreateMatrix(a))
			{
				var r2 = gpuA.SplitAtColumn(POSITION);
				using var m1 = r2.Left;
				using var m2 = r2.Right;
				gpuResults1 = m1.AsIndexable();
				gpuResults2 = m2.AsIndexable();
			}

			FloatingPointHelper.AssertEqual(gpuResults1, r.Left.AsIndexable());
			FloatingPointHelper.AssertEqual(gpuResults2, r.Right.AsIndexable());
			Cleanup();
		}

		[Test, Category("Slow")]
		public void MatrixL1Regularisation()
		{
			Load();
			var a = _cpu.CreateMatrix(6, 3, (x, y) => x * 2 + y).AsIndexable();
			const float OPERAND = 2f;
			IIndexableMatrix gpuResults;
			using (var gpuA = _cuda.CreateMatrix(a))
			{
				gpuA.L1Regularisation(OPERAND);
				gpuResults = gpuA.AsIndexable();
			}

			a.L1Regularisation(OPERAND);
			FloatingPointHelper.AssertEqual(a, gpuResults);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void MatrixColumnL2Norm()
		{
			Load();
			var a = _cpu.CreateMatrix(6, 3, (x, y) => x * 2 + y).AsIndexable();
			var r = a.ColumnL2Norm().AsIndexable();
			IIndexableVector gpuResults;
			using (var gpuA = _cuda.CreateMatrix(a))
			{
				using var norm = gpuA.ColumnL2Norm();
				gpuResults = norm.AsIndexable();
			}

			FloatingPointHelper.AssertEqual(r, gpuResults);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void MatrixRowL2Norm()
		{
			Load();
			var a = _cpu.CreateMatrix(6, 3, (x, y) => x * 2 + y).AsIndexable();
			var r = a.RowL2Norm().AsIndexable();
			IIndexableVector gpuResults;
			using (var gpuA = _cuda.CreateMatrix(a))
			{
				using var norm = gpuA.RowL2Norm();
				gpuResults = norm.AsIndexable();
			}

			FloatingPointHelper.AssertEqual(r, gpuResults);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void MatrixPointwiseDivideRows()
		{
			Load();
			var a = _cpu.CreateMatrix(6, 3, (x, y) => x * 2 + y).AsIndexable();
			var b = _cpu.CreateVector(6, i => i).AsIndexable();
			IIndexableMatrix gpuResults;
			using (var gpuA = _cuda.CreateMatrix(a))
			{
				using var gpuB = _cuda.CreateVector(b);
				gpuA.PointwiseDivideRows(gpuB);
				gpuResults = gpuA.AsIndexable();
			}

			a.PointwiseDivideRows(b);
			FloatingPointHelper.AssertEqual(a, gpuResults);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void MatrixPointwiseDivideColumns()
		{
			Load();
			var a = _cpu.CreateMatrix(6, 3, (x, y) => x * 2 + y).AsIndexable();
			var b = _cpu.CreateVector(3, i => i).AsIndexable();
			IIndexableMatrix gpuResults;
			using (var gpuA = _cuda.CreateMatrix(a))
			{
				using var gpuB = _cuda.CreateVector(b);
				gpuA.PointwiseDivideColumns(gpuB);
				gpuResults = gpuA.AsIndexable();
			}

			a.PointwiseDivideColumns(b);
			FloatingPointHelper.AssertEqual(a, gpuResults);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void MatrixDiagonal()
		{
			Load();
			var a = _cpu.CreateMatrix(6, 6, (x, y) => x * 2 + y).AsIndexable();
			var d = a.Diagonal().AsIndexable();
			IIndexableVector gpuResults;
			using (var gpuA = _cuda.CreateMatrix(a))
			{
				using var diagonal = gpuA.Diagonal();
				gpuResults = diagonal.AsIndexable();
			}

			FloatingPointHelper.AssertEqual(d, gpuResults);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void MatrixPow()
		{
			Load();
			var a = _cpu.CreateMatrix(6, 3, (x, y) => x * 2 + y).AsIndexable();
			const float OPERAND = 2.5f;
			var r = a.Pow(OPERAND).AsIndexable();
			IIndexableMatrix gpuResults;
			using (var gpuA = _cuda.CreateMatrix(a))
			{
				using var pow = gpuA.Pow(OPERAND);
				gpuResults = pow.AsIndexable();
			}

			FloatingPointHelper.AssertEqual(r, gpuResults);
			Cleanup();
		}

		//[Test, Category("Slow")]
		//public void MatrixUpdateRow()
		//{
		//    var a = _cpu.CreateMatrix(3, 7, (x, y) => x * 2 + y).AsIndexable();
		//    var r = _cpu.CreateVector(2, x => -1f).AsIndexable();

		//    IIndexableMatrix gpuResults;
		//    using (var gpuA = _cuda.CreateMatrix(a)) {
		//        gpuA.UpdateRow(2, r, 3);
		//        gpuResults = gpuA.AsIndexable();
		//    }

		//    a.UpdateRow(2, r, 3);
		//    FloatingPointHelper.AssertEqual(a, gpuResults);
		//}

		//[Test, Category("Slow")]
		//public void MatrixUpdateColumn()
		//{
		//    var a = _cpu.CreateMatrix(13, 17, (x, y) => x * 2 + y).AsIndexable();
		//    var r = _cpu.CreateVector(2, x => -1f).AsIndexable();

		//    IIndexableMatrix gpuResults;
		//    using (var gpuA = _cuda.CreateMatrix(a)) {
		//        gpuA.UpdateColumn(2, r, 3);
		//        gpuResults = gpuA.AsIndexable();
		//    }

		//    a.UpdateColumn(2, r, 3);
		//    FloatingPointHelper.AssertEqual(a, gpuResults);
		//}

		[Test, Category("Slow")]
		public void MatrixGetRowSegment()
		{
			Load();
			var a = _cpu.CreateMatrix(12, 18, (x, y) => x * 2 + y).AsIndexable();
			var r = a.GetRowSegment(1, 2, 5).AsIndexable();
			IIndexableVector gpuResults;
			using (var gpuA = _cuda.CreateMatrix(a))
			{
				gpuResults = gpuA.GetRowSegment(1, 2, 5).AsIndexable();
			}

			FloatingPointHelper.AssertEqual(r, gpuResults);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void MatrixGetColumnSegment()
		{
			Load();
			var a = _cpu.CreateMatrix(9, 8, (x, y) => (x + 1) * (y + 1)).AsIndexable();
			var r = a.GetColumnSegment(1, 2, 5).AsIndexable();
			IIndexableVector gpuResults;
			using (var gpuA = _cuda.CreateMatrix(a))
			{
				gpuResults = gpuA.GetColumnSegment(1, 2, 5).AsIndexable();
			}

			FloatingPointHelper.AssertEqual(r, gpuResults);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void MatrixConstrain()
		{
			Load();
			var distribution = new Normal(0, 5);
			var a = _cpu.CreateMatrix(100, 100, (x, y) => Convert.ToSingle(distribution.Sample())).
				AsIndexable();
			IIndexableMatrix gpuResults;
			using (var gpuA = _cuda.CreateMatrix(a))
			{
				gpuA.Constrain(-2f, 2f);
				gpuResults = gpuA.AsIndexable();
			}

			a.Constrain(-2f, 2f);
			FloatingPointHelper.AssertEqual(a, gpuResults);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void TestIdentity()
		{
			Load();
			var a = _cpu.CreateIdentityMatrix(1000).AsIndexable();
			IIndexableMatrix a2;
			using (var gpuA = _cuda.CreateIdentityMatrix(1000))
				a2 = gpuA.AsIndexable();
			FloatingPointHelper.AssertEqual(a, a2);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void MatrixSvd()
		{
			Load();
			var a = _cpu.CreateZeroMatrix(2, 2).AsIndexable();
			a[0, 0] = 4;
			a[0, 1] = 7;
			a[1, 0] = 2;
			a[1, 1] = 6;
			var svd = a.Svd();
			var cpuU = svd.U.AsIndexable();
			var cpuVT = svd.VT.AsIndexable();
			var cpuS = svd.S.AsIndexable();
			IIndexableMatrix gpuU, gpuVT;
			IIndexableVector gpuS;
			using (var gpuA = _cuda.CreateMatrix(a))
			{
				var gpuSvd = gpuA.Svd();
				gpuU = gpuSvd.U.AsIndexable();
				gpuVT = gpuSvd.VT.AsIndexable();
				gpuS = gpuSvd.S.AsIndexable();
			}

			FloatingPointHelper.AssertEqual(cpuU, gpuU);
			FloatingPointHelper.AssertEqual(cpuVT, gpuVT);
			FloatingPointHelper.AssertEqual(cpuS, gpuS);
			Cleanup();
		}

		[Test, Category("Slow")]
		public void MatrixToVector()
		{
			Load();
			var matrix = _cpu.CreateMatrix(3, 4, (x, y) => (x + 1) * (y + 1)).AsIndexable();
			var vector = matrix.ReshapeAsVector();
			var matrix2 = vector.ReshapeAsMatrix(3, 4);
			FloatingPointHelper.AssertEqual(matrix.AsIndexable(), matrix2.AsIndexable());
			using var gpuMatrix = _cuda.CreateMatrix(matrix.AsIndexable());
			using var gpuVector = gpuMatrix.ReshapeAsVector();
			using var gpuMatrix2 = gpuVector.ReshapeAsMatrix(3, 4);
			FloatingPointHelper.AssertEqual(gpuMatrix.AsIndexable(), gpuMatrix2.AsIndexable());
			Cleanup();
		}

		public static void Cleanup()
		{
			_cuda.Dispose();
			_cpu.Dispose();
		}
	}
}