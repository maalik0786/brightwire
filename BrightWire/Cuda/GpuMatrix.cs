using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using BrightWire.Models;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.CudaBlas;
using Math = System.Math;

namespace BrightWire.Cuda
{
	/// <summary>
	/// GPU backed matrix
	/// </summary>
	class GpuMatrix : IMatrix, IHaveDeviceMemory
	{
		readonly CudaProvider _cuda;
		bool _disposed;
#if DEBUG
		static int _gid;
		static int _GetNextIndex() => Interlocked.Increment(ref _gid);
		readonly int _id = _GetNextIndex();
		public static int _badAlloc = -1;
		public static int _badDispose = -1;

		public bool IsValid => !_disposed;
#else
        public bool IsValid => true;
#endif

		public GpuMatrix(CudaProvider cuda, int rows, int columns, IDeviceMemoryPtr data, bool isOwner)
		{
			Debug.Assert(rows * columns == data.Size);
			_cuda = cuda;
			RowCount = rows;
			ColumnCount = columns;
			Memory = data;
			cuda.Register(this);
#if DEBUG
			if (_id == _badAlloc)
				Debugger.Break();
#endif
		}

#if DEBUG
		~GpuMatrix()
		{
			if (!_disposed)
				Debug.WriteLine("\tMatrix {0} was not disposed !!", _id);
		}
#endif

		protected virtual void Dispose(bool disposing)
		{
#if DEBUG
			if (_id == _badDispose)
				Debugger.Break();
#endif
			if (disposing && !_disposed)
			{
				Memory.Free();
				_disposed = true;
			}
		}

		public void Dispose()
		{
			Dispose(true);
#if DEBUG
			GC.SuppressFinalize(this);
#endif
		}

		public override string ToString()
		{
			return AsIndexable().ToString();
		}

		public int ColumnCount { get; }
		public int RowCount { get; }
		public IDeviceMemoryPtr Memory { get; }

		public IMatrix Add(IMatrix matrix)
		{
			Debug.Assert(IsValid && matrix.IsValid);
			var other = (GpuMatrix)matrix;
			Debug.Assert(other.RowCount == RowCount && other.ColumnCount == ColumnCount);
			var ret = _cuda.Allocate(other.Memory.Size);
			ret.CopyToDevice(other.Memory);
			_cuda.Blas.Axpy(1.0f, Memory.DeviceVariable, 1, ret.DeviceVariable, 1);
			return new GpuMatrix(_cuda, RowCount, ColumnCount, ret, true);
		}

		public void AddInPlace(IMatrix matrix, float coefficient1 = 1, float coefficient2 = 1)
		{
			Debug.Assert(IsValid && matrix.IsValid);
			var other = (GpuMatrix)matrix;
			Debug.Assert(other.RowCount == RowCount && other.ColumnCount == ColumnCount);
			_cuda.AddInPlace(Memory, other.Memory, RowCount * ColumnCount, coefficient1, coefficient2);
		}

		public void AddToEachColumn(IVector vector)
		{
			Debug.Assert(IsValid && vector.IsValid);
			var other = (GpuVector)vector;
			_cuda.AddToEachColumn(Memory, other.Memory, RowCount, ColumnCount);
		}

		public void AddToEachRow(IVector vector)
		{
			Debug.Assert(IsValid && vector.IsValid);
			var other = (GpuVector)vector;
			_cuda.AddToEachRow(Memory, other.Memory, RowCount, ColumnCount);
		}

		public IIndexableMatrix AsIndexable()
		{
			Debug.Assert(IsValid);
			var data = new float[RowCount * ColumnCount];
			Memory.CopyToHost(data);
			return _cuda.NumericsProvider.
				CreateMatrix(RowCount, ColumnCount, (j, k) => data[k * RowCount + j]).AsIndexable();
		}

		public void Clear()
		{
			Debug.Assert(IsValid);
			Memory.DeviceVariable.Memset(0);
		}

		public void ClearColumns(IReadOnlyList<int> indices)
		{
			Debug.Assert(IsValid);
			foreach (var item in indices)
				_cuda.MemClear(Memory, RowCount, item * RowCount);
		}

		public void ClearRows(IReadOnlyList<int> indices)
		{
			Debug.Assert(IsValid);
			foreach (var item in indices)
				_cuda.MemClear(Memory, ColumnCount, item, RowCount);
		}

		public IMatrix Clone()
		{
			Debug.Assert(IsValid);
			var ret = _cuda.Allocate(RowCount * ColumnCount);
			ret.CopyToDevice(Memory);
			return new GpuMatrix(_cuda, RowCount, ColumnCount, ret, true);
		}

		public IVector Column(int index)
		{
			Debug.Assert(IsValid);
			var ptr = _cuda.OffsetByBlock(Memory, index, RowCount);
			return new GpuVector(_cuda, ptr, false);
		}

		public IVector ColumnL2Norm()
		{
			Debug.Assert(IsValid);
			var norm = new List<float>();
			for (var i = 0; i < ColumnCount; i++)
			{
				using var col = Column(i);
				norm.Add(col.L2Norm());
			}

			return _cuda.CreateVector(norm.Count, x => norm[x]);
		}

		public IVector ColumnSums()
		{
			Debug.Assert(IsValid);
			return new GpuVector(_cuda, _cuda.SumColumns(Memory, RowCount, ColumnCount), true);
		}

		public IMatrix ConcatColumns(IMatrix bottom)
		{
			Debug.Assert(IsValid && bottom.IsValid);
			var t = this;
			var b = (GpuMatrix)bottom;
			Debug.Assert(ColumnCount == bottom.ColumnCount);
			var size = t.RowCount + b.RowCount;
			var ret = _cuda.Allocate(size * t.ColumnCount);
			_cuda.ConcatColumns(t.Memory, b.Memory, ret, size, t.ColumnCount, t.RowCount, b.RowCount);
			return new GpuMatrix(_cuda, size, t.ColumnCount, ret, true);
		}

		public IMatrix ConcatRows(IMatrix right)
		{
			Debug.Assert(IsValid && right.IsValid);
			var t = this;
			var b = (GpuMatrix)right;
			Debug.Assert(RowCount == right.RowCount);
			var size = t.ColumnCount + b.ColumnCount;
			var ret = _cuda.Allocate(t.RowCount * size);
			_cuda.ConcatRows(t.Memory, b.Memory, ret, t.RowCount, size, t.ColumnCount);
			return new GpuMatrix(_cuda, t.RowCount, size, ret, true);
		}

		public void Constrain(float min, float max)
		{
			Debug.Assert(IsValid);
			_cuda.Constrain(Memory, RowCount * ColumnCount, min, max);
		}

		public IVector Diagonal()
		{
			Debug.Assert(IsValid);
			var ret = _cuda.Diagonal(Memory, RowCount, ColumnCount);
			return new GpuVector(_cuda, ret, true);
		}

		public IVector GetColumnSegment(int columnIndex, int rowIndex, int length)
		{
			Debug.Assert(IsValid);
			var ret = _cuda.Allocate(length);
			ret.DeviceVariable.CopyToDevice(Memory.DeviceVariable,
				((columnIndex * RowCount) + rowIndex) * CudaProvider.FLOAT_SIZE, 0,
				length * CudaProvider.FLOAT_SIZE);
			return new GpuVector(_cuda, ret, true);
		}

		public IMatrix GetNewMatrixFromColumns(IReadOnlyList<int> columnIndices)
		{
			Debug.Assert(IsValid);
			int offset = 0;
			var ret = _cuda.Allocate(RowCount * columnIndices.Count);
			foreach (var item in columnIndices)
			{
				ret.DeviceVariable.CopyToDevice(Memory.DeviceVariable,
					item * RowCount * CudaProvider.FLOAT_SIZE, offset * CudaProvider.FLOAT_SIZE,
					RowCount * CudaProvider.FLOAT_SIZE);
				offset += RowCount;
			}

			return new GpuMatrix(_cuda, RowCount, columnIndices.Count, ret, true);
		}

		public IMatrix GetNewMatrixFromRows(IReadOnlyList<int> rowIndices)
		{
			Debug.Assert(IsValid);
			int offset = 0;
			var ret = _cuda.Allocate(ColumnCount * rowIndices.Count);
			foreach (var item in rowIndices)
			{
				CudaBlasNativeMethods.cublasScopy_v2(_cuda.Blas.CublasHandle, n: ColumnCount,
					x: Memory.DevicePointer + (item * CudaProvider.FLOAT_SIZE), incx: RowCount,
					y: ret.DevicePointer + (offset * CudaProvider.FLOAT_SIZE), incy: rowIndices.Count);
				offset += 1;
			}

			return new GpuMatrix(_cuda, rowIndices.Count, ColumnCount, ret, true);
		}

		public IVector GetRowSegment(int rowIndex, int columnIndex, int length)
		{
			Debug.Assert(IsValid);
			int offset = (rowIndex + (columnIndex * RowCount)) * CudaProvider.FLOAT_SIZE;
			var ret = _cuda.Allocate(length);
			CudaBlasNativeMethods.cublasScopy_v2(_cuda.Blas.CublasHandle, length,
				Memory.DevicePointer + offset, RowCount, ret.DevicePointer, 1);
			return new GpuVector(_cuda, ret, true);
		}

		public void L1Regularisation(float coefficient)
		{
			Debug.Assert(IsValid);
			_cuda.L1Regularisation(Memory, RowCount * ColumnCount, coefficient);
		}

		public IMatrix LeakyReluActivation()
		{
			Debug.Assert(IsValid);
			var ret = _cuda.LeakyRELU(Memory, RowCount * ColumnCount);
			return new GpuMatrix(_cuda, RowCount, ColumnCount, ret, true);
		}

		public IMatrix LeakyReluDerivative()
		{
			Debug.Assert(IsValid);
			var ret = _cuda.LeakyRELUDerivative(Memory, RowCount * ColumnCount);
			return new GpuMatrix(_cuda, RowCount, ColumnCount, ret, true);
		}

		public void Multiply(float scalar)
		{
			Debug.Assert(IsValid);
			_cuda.Blas.Scale(scalar, Memory.DeviceVariable, 1);
		}

		public IMatrix Multiply(IMatrix matrix)
		{
			Debug.Assert(IsValid && matrix.IsValid);
			var other = (GpuMatrix)matrix;
			Debug.Assert(ColumnCount == other.RowCount);
			var ret = _cuda.Allocate(RowCount * other.ColumnCount);
			int rowsA = RowCount, columnsArowsB = ColumnCount, columnsB = other.ColumnCount;
			float alpha = 1.0f, beta = 0.0f;
			CudaBlasNativeMethods.cublasSgemm_v2(_cuda.Blas.CublasHandle, Operation.NonTranspose,
				Operation.NonTranspose, rowsA, columnsB, columnsArowsB, ref alpha, Memory.DevicePointer,
				rowsA, other.Memory.DevicePointer, columnsArowsB, ref beta, ret.DevicePointer, rowsA);
			return new GpuMatrix(_cuda, RowCount, other.ColumnCount, ret, true);
		}

		public IMatrix PointwiseDivide(IMatrix matrix)
		{
			Debug.Assert(IsValid && matrix.IsValid);
			var other = (GpuMatrix)matrix;
			Debug.Assert(other.RowCount == RowCount && other.ColumnCount == ColumnCount);
			var size = RowCount * ColumnCount;
			var ret = _cuda.PointwiseDivide(Memory, other.Memory, size);
			return new GpuMatrix(_cuda, RowCount, ColumnCount, ret, true);
		}

		public void PointwiseDivideColumns(IVector vector)
		{
			Debug.Assert(IsValid && vector.IsValid);
			var other = (GpuVector)vector;
			_cuda.PointwiseDivideColumns(Memory, other.Memory, RowCount, ColumnCount);
		}

		public void PointwiseDivideRows(IVector vector)
		{
			Debug.Assert(IsValid && vector.IsValid);
			var other = (GpuVector)vector;
			_cuda.PointwiseDivideRows(Memory, other.Memory, RowCount, ColumnCount);
		}

		public IMatrix PointwiseMultiply(IMatrix matrix)
		{
			Debug.Assert(IsValid && matrix.IsValid);
			var other = (GpuMatrix)matrix;
			Debug.Assert(other.RowCount == RowCount && other.ColumnCount == ColumnCount);
			var size = RowCount * ColumnCount;
			var ret = _cuda.PointwiseMultiply(Memory, other.Memory, size);
			return new GpuMatrix(_cuda, RowCount, ColumnCount, ret, true);
		}

		public IMatrix Pow(float power)
		{
			Debug.Assert(IsValid);
			var ret = _cuda.Pow(Memory, RowCount * ColumnCount, power);
			return new GpuMatrix(_cuda, RowCount, ColumnCount, ret, true);
		}

		public IMatrix ReluActivation()
		{
			Debug.Assert(IsValid);
			return new GpuMatrix(_cuda, RowCount, ColumnCount, _cuda.RELU(Memory, RowCount * ColumnCount),
				true);
		}

		public IMatrix ReluDerivative()
		{
			Debug.Assert(IsValid);
			return new GpuMatrix(_cuda, RowCount, ColumnCount,
				_cuda.RELUDerivative(Memory, RowCount * ColumnCount), true);
		}

		public IVector Row(int index)
		{
			Debug.Assert(IsValid);
			var ret = _cuda.Allocate(ColumnCount);
			int offset = index * CudaProvider.FLOAT_SIZE;
			CudaBlasNativeMethods.cublasScopy_v2(_cuda.Blas.CublasHandle, ColumnCount,
				Memory.DevicePointer + offset, RowCount, ret.DevicePointer, 1);
			return new GpuVector(_cuda, ret, true);
		}

		public IVector RowL2Norm()
		{
			Debug.Assert(IsValid);
			var norm = new List<float>();
			for (var i = 0; i < RowCount; i++)
			{
				using var row = Row(i);
				norm.Add(row.L2Norm());
			}

			return _cuda.CreateVector(norm.Count, x => norm[x]);
		}

		public IVector RowSums()
		{
			Debug.Assert(IsValid);
			return new GpuVector(_cuda, _cuda.SumRows(Memory, RowCount, ColumnCount), true);
		}

		public IMatrix SigmoidActivation()
		{
			Debug.Assert(IsValid);
			return new GpuMatrix(_cuda, RowCount, ColumnCount,
				_cuda.Sigmoid(Memory, RowCount * ColumnCount), true);
		}

		public IMatrix SigmoidDerivative()
		{
			Debug.Assert(IsValid);
			return new GpuMatrix(_cuda, RowCount, ColumnCount,
				_cuda.SigmoidDerivative(Memory, RowCount * ColumnCount), true);
		}

		public IMatrix SoftmaxActivation()
		{
			Debug.Assert(IsValid);
			var rowOutput = new List<GpuVector>();
			for (var i = 0; i < RowCount; i++)
			{
				using var row = Row(i);
				rowOutput.Add(row.Softmax() as GpuVector);
			}

			var ret = _cuda.Allocate(RowCount * ColumnCount);
			for (var i = 0; i < RowCount; i++)
			{
				using var row = rowOutput[i];
				ret.DeviceVariable.CopyToDevice(row.Memory.DeviceVariable, 0,
					ColumnCount * i * CudaProvider.FLOAT_SIZE, ColumnCount * CudaProvider.FLOAT_SIZE);
			}

			using var temp = new GpuMatrix(_cuda, ColumnCount, RowCount, ret, true);
			return temp.Transpose();
		}

		public (IMatrix Top, IMatrix Bottom) SplitAtRow(int rowIndex)
		{
			Debug.Assert(IsValid);
			int size = RowCount - rowIndex;
			var ret1 = _cuda.Allocate(rowIndex * ColumnCount);
			var ret2 = _cuda.Allocate(size * ColumnCount);
			_cuda.SplitColumns(Memory, ret1, ret2, RowCount, ColumnCount, rowIndex);
			return (new GpuMatrix(_cuda, rowIndex, ColumnCount, ret1, true),
				new GpuMatrix(_cuda, size, ColumnCount, ret2, true));
		}

		public (IMatrix Left, IMatrix Right) SplitAtColumn(int columnIndex)
		{
			Debug.Assert(IsValid);
			int size = ColumnCount - columnIndex;
			var ret1 = _cuda.Allocate(RowCount * columnIndex);
			var ret2 = _cuda.Allocate(RowCount * size);
			_cuda.SplitRows(Memory, ret1, ret2, RowCount, ColumnCount, columnIndex);
			return (new GpuMatrix(_cuda, RowCount, columnIndex, ret1, true),
				new GpuMatrix(_cuda, RowCount, size, ret2, true));
		}

		public IMatrix Sqrt(float valueAdjustment = 1e-8f)
		{
			Debug.Assert(IsValid);
			var size = RowCount * ColumnCount;
			var ret = _cuda.Sqrt(Memory, size, valueAdjustment);
			return new GpuMatrix(_cuda, RowCount, ColumnCount, ret, true);
		}

		public IMatrix Subtract(IMatrix matrix)
		{
			Debug.Assert(IsValid && matrix.IsValid);
			var other = (GpuMatrix)matrix;
			Debug.Assert(other.RowCount == RowCount && other.ColumnCount == ColumnCount);
			var ret = _cuda.Allocate(Memory.Size);
			ret.CopyToDevice(Memory);
			_cuda.Blas.Axpy(-1.0f, other.Memory.DeviceVariable, 1, ret.DeviceVariable, 1);
			return new GpuMatrix(_cuda, RowCount, ColumnCount, ret, true);
		}

		public void SubtractInPlace(IMatrix matrix, float coefficient1 = 1, float coefficient2 = 1)
		{
			Debug.Assert(IsValid && matrix.IsValid);
			var other = (GpuMatrix)matrix;
			Debug.Assert(other.RowCount == RowCount && other.ColumnCount == ColumnCount);
			_cuda.SubtractInPlace(Memory, other.Memory, RowCount * ColumnCount, coefficient1,
				coefficient2);
		}

		public IMatrix TanhActivation()
		{
			Debug.Assert(IsValid);
			return new GpuMatrix(_cuda, RowCount, ColumnCount, _cuda.TanH(Memory, RowCount * ColumnCount),
				true);
		}

		public IMatrix TanhDerivative()
		{
			Debug.Assert(IsValid);
			return new GpuMatrix(_cuda, RowCount, ColumnCount,
				_cuda.TanHDerivative(Memory, RowCount * ColumnCount), true);
		}

		public IMatrix Transpose()
		{
			Debug.Assert(IsValid);
			var ret = _cuda.Allocate(RowCount * ColumnCount);
			float alpha = 1.0f, beta = 0.0f;
			CudaBlasNativeMethods.cublasSgeam(_cuda.Blas.CublasHandle, Operation.Transpose,
				Operation.NonTranspose, ColumnCount, RowCount, ref alpha, Memory.DevicePointer, RowCount,
				ref beta, new CUdeviceptr(0), ColumnCount, ret.DevicePointer, ColumnCount);
			return new GpuMatrix(_cuda, ColumnCount, RowCount, ret, true);
		}

		public IMatrix TransposeAndMultiply(IMatrix matrix)
		{
			Debug.Assert(IsValid && matrix.IsValid);
			var other = (GpuMatrix)matrix;
			Debug.Assert(ColumnCount == other.ColumnCount);
			var ret = _cuda.Allocate(RowCount * other.RowCount);
			int rowsA = RowCount, columnsArowsB = ColumnCount, rowsB = other.RowCount;
			float alpha = 1.0f, beta = 0.0f;
			CudaBlasNativeMethods.cublasSgemm_v2(_cuda.Blas.CublasHandle, Operation.NonTranspose,
				Operation.Transpose, rowsA, rowsB, columnsArowsB, ref alpha, Memory.DevicePointer, rowsA,
				other.Memory.DevicePointer, rowsB, ref beta, ret.DevicePointer, rowsA);
			return new GpuMatrix(_cuda, RowCount, other.RowCount, ret, true);
		}

		public IMatrix TransposeThisAndMultiply(IMatrix matrix)
		{
			Debug.Assert(IsValid && matrix.IsValid);
			var other = (GpuMatrix)matrix;
			Debug.Assert(RowCount == other.RowCount);
			var ret = _cuda.Allocate(ColumnCount * other.ColumnCount);
			int rowsA = RowCount, columnsA = ColumnCount, columnsB = other.ColumnCount,
				rowsB = other.RowCount;
			float alpha = 1.0f, beta = 0.0f;
			CudaBlasNativeMethods.cublasSgemm_v2(_cuda.Blas.CublasHandle, Operation.Transpose,
				Operation.NonTranspose, columnsA, columnsB, rowsB, ref alpha, Memory.DevicePointer, rowsA,
				other.Memory.DevicePointer, rowsB, ref beta, ret.DevicePointer, columnsA);
			return new GpuMatrix(_cuda, ColumnCount, other.ColumnCount, ret, true);
		}

		public FloatMatrix Data
		{
			get => AsIndexable().Data;
			set
			{
				Debug.Assert(IsValid);
				var buffer = new float[RowCount * ColumnCount];
				Memory.CopyToHost(buffer);
				var rowCount = value.Row.Length;
				for (var i = 0; i < rowCount && i < RowCount; i++)
				{
					var row = value.Row[i];
					if (row.Data != null)
					{
						var data2 = row.Data;
						var columnCount = data2.Length;
						for (var j = 0; j < columnCount && j < ColumnCount; j++)
						{
							buffer[j * RowCount + i] = data2[j];
						}
					}
				}

				Memory.CopyToDevice(buffer);
			}
		}

		public IMatrix Multiply(IVector vector)
		{
			Debug.Assert(IsValid && vector.IsValid);
			using var column = vector.ReshapeAsColumnMatrix();
			return Multiply(column);
		}

		public (IMatrix U, IVector S, IMatrix VT) Svd()
		{
			Debug.Assert(IsValid);
			var solver = _cuda.Solver;

			// find the size of the required buffer
			var bufferSize = solver.GesvdBufferSizeFloat(RowCount, ColumnCount);
			var mn = Math.Min(RowCount, ColumnCount);

			// allocate output buffers
			var s = _cuda.Allocate(mn);
			var u = _cuda.Allocate(RowCount * RowCount);
			var vt = _cuda.Allocate(ColumnCount * ColumnCount);

			// call cusolver to find the SVD
			try
			{
				var buffer = _cuda.Allocate(bufferSize);
				var rwork = _cuda.Allocate(mn);
				var a = _cuda.Allocate(RowCount * ColumnCount);
				try
				{
					using var devInfo = new CudaDeviceVariable<int>(1);
					a.CopyToDevice(Memory);
					solver.Gesvd('A', 'A', RowCount, ColumnCount, a.DeviceVariable, RowCount,
						s.DeviceVariable, u.DeviceVariable, RowCount, vt.DeviceVariable, ColumnCount,
						buffer.DeviceVariable, bufferSize, rwork.DeviceVariable, devInfo);
					return (new GpuMatrix(_cuda, RowCount, RowCount, u, true), new GpuVector(_cuda, s, true),
						new GpuMatrix(_cuda, ColumnCount, ColumnCount, vt, true));
				}
				finally
				{
					buffer.Free();
					rwork.Free();
					a.Free();
				}
			}
			catch
			{
				s.Free();
				u.Free();
				vt.Free();
				throw;
			}
		}

		public IVector ReshapeAsVector()
		{
			Debug.Assert(IsValid);
			return new GpuVector(_cuda, Memory, false);
		}

		public I3DTensor ReshapeAs3DTensor(int rows, int columns)
		{
			Debug.Assert(IsValid && rows * columns == RowCount);
			return new Gpu3DTensor(_cuda, rows, columns, ColumnCount, Memory, false);
		}

		public I4DTensor ReshapeAs4DTensor(int rows, int columns, int depth)
		{
			Debug.Assert(IsValid && rows * columns * depth == RowCount);
			return new Gpu4DTensor(_cuda, rows, columns, depth, ColumnCount, Memory, false);
		}

		public float GetAt(int row, int column)
		{
			return Memory.DeviceVariable[column * RowCount + row];
		}

		public void SetAt(int row, int column, float value)
		{
			Memory.DeviceVariable[column * RowCount + row] = value;
		}

		public IReadOnlyList<IVector> ColumnVectors()
		{
			var ret = new List<IVector>();
			for (var i = 0; i < ColumnCount; i++)
				ret.Add(Column(i));
			return ret;
		}

		public IReadOnlyList<IVector> RowVectors()
		{
			var ret = new List<IVector>();
			for (var i = 0; i < RowCount; i++)
				ret.Add(Row(i));
			return ret;
		}
	}
}