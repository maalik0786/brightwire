﻿using BrightWire.Helper;
using BrightWire.Models;
using MathNet.Numerics.LinearAlgebra.Single;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;

namespace BrightWire.LinearAlgebra
{
    internal class Cpu3DTensor : IIndexable3DTensor
    {
        readonly CpuMatrix[] _data;
        readonly int _rows, _columns, _depth;

        public Cpu3DTensor(int rows, int columns, int depth)
        {
            _rows = rows;
            _columns = columns;
            _depth = depth;
            _data = Enumerable.Range(0, depth)
                .Select(i => new CpuMatrix(DenseMatrix.Create(_rows, _columns, 0f)))
                .ToArray()
            ;
        }

        public Cpu3DTensor(IReadOnlyList<IMatrix> matrixList)
        {
            var first = matrixList.First();
            Debug.Assert(matrixList.All(m => m.RowCount == first.RowCount && m.ColumnCount == first.ColumnCount));
            _rows = first.RowCount;
            _columns = first.ColumnCount;
            _depth = matrixList.Count;
            _data = matrixList.Cast<CpuMatrix>().ToArray();
        }

        public void Dispose()
        {
            // nop
        }

        public int AddRef()
        {
            // nop
            return 1;
        }

        public int Release()
        {
            // nop
            return 1;
        }
        public float this[int row, int column, int depth]
        {
            get
            {
                return _data[depth][row, column];
            }

            set
            {
                _data[depth][row, column] = value;
            }
        }

        public FloatTensor Data
        {
            get
            {
                return new FloatTensor {
                    Matrix = _data.Select(m => m.Data).ToArray()
                };
            }
            set
            {
                var matrixList = value.Matrix;
                var matrixCount = matrixList.Length;
                for (var i = 0; i < matrixCount && i < _data.Length; i++) {
                    var matrix = matrixList[i];
                    if (matrix.Row != null)
                        _data[i].Data = matrix;
                }
            }
        }

        public IReadOnlyList<IIndexableMatrix> Matrix => _data;
        public int Depth => _depth;
        public int RowCount => _rows;
        public int ColumnCount => _columns;
        public IMatrix GetDepthSlice(int depth) => _data[depth];
        public IIndexable3DTensor AsIndexable() => this;
        public IReadOnlyList<IMatrix> DepthSlices => _data;

        public I3DTensor AddPadding(int padding)
        {
            var newRows = _rows + padding * 2;
            var newColumns = _columns + padding * 2;
            var ret = new Cpu3DTensor(newRows, newColumns, _depth);

            for (var k = 0; k < _depth; k++) {
                for (var j = 0; j < newRows; j++) {
                    for (var i = 0; i < newColumns; i++) {
                        if (i < padding || j < padding)
                            continue;
                        else if (i >= newRows - padding || j >= newColumns - padding)
                            continue;
                        ret[i, j, k] = this[i - padding, j - padding, k];
                    }
                }
            }
            return ret;
        }

        public I3DTensor RemovePadding(int padding)
        {
            var newRows = _rows - padding * 2;
            var newColumns = _columns - padding * 2;
            var ret = new Cpu3DTensor(newRows, newColumns, _depth);
            for (var k = 0; k < _depth; k++) {
                for (var j = 0; j < newRows; j++) {
                    for (var i = 0; i < newColumns; i++) {
                        ret[i, j, k] = this[i + padding, j + padding, k];
                    }
                }
            }
            return ret;
        }

        public IMatrix Im2Col(int filterWidth, int filterHeight, int stride)
        {
            int y = 0, x = 0;
            var rowList = new List<List<float>>();
            var convolutions = ConvolutionHelper.Default(ColumnCount, RowCount, filterWidth, filterHeight, stride);

            foreach (var filter in convolutions) {
                var row = new List<float>();
                for (var k = 0; k < _depth; k++) {
                    foreach (var item in filter)
                        row.Add(this[item.Y, item.X, k]);
                }
                rowList.Add(row);
            }
            var firstRow = rowList.First();
            return new CpuMatrix(DenseMatrix.Create(rowList.Count, firstRow.Count, (i, j) => rowList[i][j]));
        }

        public IVector ConvertToVector()
        {
            var vectorList = _data.Select(m => m.ConvertInPlaceToVector().AsIndexable()).ToArray();
            var size = _rows * _columns;
            var ret = DenseVector.Create(_depth * size, i => {
                var offset = i / size;
                var index = i % size;
                return vectorList[offset][index];
            });
            return new CpuVector(ret);
        }

        public IMatrix ConvertToMatrix()
        {
            var ret = DenseMatrix.Create(_rows * _columns, _depth, (i, j) => {
                var matrix = _data[j];
                var x = i / _columns;
                var y = i % _columns;
                return matrix[x, y];
            });
            return new CpuMatrix(ret);
        }

        public (I3DTensor Result, IReadOnlyList<(object X, object Y)> Index) MaxPool(
            int filterWidth, 
            int filterHeight, 
            int stride,
            bool calculateIndex)
        {
            var newColumns = (ColumnCount - filterWidth) / stride + 1;
            var newRows = (RowCount - filterHeight) / stride + 1;
            var matrixList = new List<CpuMatrix>();
            var indexList = calculateIndex ? new List<(object X, object Y)>() : null;
            var posList = ConvolutionHelper.Default(ColumnCount, RowCount, filterWidth, filterHeight, stride);

            for (var k = 0; k < Depth; k++) {
                var matrix = _data[k];
                var xIndex = new int[newColumns * newRows];
                var yIndex = new int[newColumns * newRows];
                var layer = new CpuMatrix(DenseMatrix.Create(newRows, newColumns, 0f));
                
                foreach(var item in posList) {
                    var first = item.First();
                    var targetX = first.X / stride;
                    var targetY = first.Y / stride;
                    var maxVal = float.MinValue;
                    var bestX = -1;
                    var bestY = -1;
                    foreach (var pos in item) {
                        var val = matrix[pos.Y, pos.X];
                        if (val > maxVal || bestX == -1) {
                            bestX = pos.X;
                            bestY = pos.Y;
                            maxVal = val;
                        }
                    }
                    var index = targetX * newRows + targetY;
                    xIndex[index] = bestX;
                    yIndex[index] = bestY;
                    layer[targetY, targetX] = maxVal;
                }
                matrixList.Add(layer);
                indexList?.Add((xIndex, yIndex));
            }
            return (new Cpu3DTensor(matrixList), indexList);
        }

        public I3DTensor ReverseMaxPool(int rows, int columns, IReadOnlyList<(object X, object Y)> indexList)
        {
            var matrixList = new List<DenseMatrix>();
            for (var z = 0; z < Depth; z++) {
                var source = GetDepthSlice(z).AsIndexable();
                var newRows = source.RowCount;
                var index = indexList[z];
                int[] indexX = (int[])index.X;
                int[] indexY = (int[])index.Y;
                var target = DenseMatrix.Create(rows, columns, 0f);
                
                for(int i = 0, len = indexX.Length; i < len; i++) {
                    var x = indexX[i];
                    var y = indexY[i];
                    var sourceX = i / newRows;
                    var sourceY = i % newRows;
                    target[y, x] = source[sourceY, sourceX];
                }

                matrixList.Add(target);
            }
            return new Cpu3DTensor(matrixList.Select(m => new CpuMatrix(m)).ToList());
        }

        //public (IMatrix WeightUpdate, IVector BiasUpdate) CalculateWeightUpdate(IMatrix im2Col)
        //{
        //    var multiplyWith = DenseMatrix.Create(RowCount * ColumnCount, Depth, 0f);
        //    var biasUpdate = new float[Depth];

        //    for (var k = 0; k < Depth; k++) {
        //        var total = 0f;
        //        int count = 0;
        //        var slice = GetDepthSlice(k).AsIndexable();
        //        for (var x = 0; x < slice.ColumnCount; x++) {
        //            for (var y = 0; y < slice.RowCount; y++) {
        //                var value = slice[x, y];
        //                multiplyWith[x * slice.RowCount + y, k] = value;
        //                total += value;
        //                ++count;
        //            }
        //        }
        //        biasUpdate[k] = total / count;
        //    }

        //    var delta = im2Col.TransposeThisAndMultiply(new CpuMatrix(multiplyWith));
        //    var biasUpdateVector = new CpuVector(biasUpdate);
        //    return (delta, biasUpdateVector);
        //}

        public I3DTensor CalculatePreviousError(IMatrix filterMatrix, int inputHeight, int inputWidth, int inputDepth, int padding, int filterHeight, int filterWidth, int stride)
        {
            var filters = filterMatrix.AsIndexable().Columns.ToList();
            var columns = inputWidth + padding * 2;
            var rows = inputHeight + padding * 2;
            var matrixList = Enumerable.Range(0, inputDepth)
                .Select(i => DenseMatrix.Create(rows, columns, 0f))
                .ToList()
            ;
            var convolutions = ConvolutionHelper.Default(columns, rows, filterHeight, filterHeight, stride);

            for (var k = 0; k < Depth; k++) {
                var slice = GetDepthSlice(k).AsIndexable();
                var filterList = filters[k]
                    .Split(inputDepth)
                    .Select(f => f.Rotate(f.Count / filterWidth).AsIndexable())
                    .ToList()
                ;

                foreach (var convolution in convolutions) {
                    var first = convolution.First();
                    var error = slice[first.X / stride, first.Y / stride];
                    if (error != 0) {
                        foreach (var item in convolution) {
                            var i = item.X - first.X;
                            var j = item.Y - first.Y;
                            var filterIndex = i * filterHeight + j;
                            for (var z = 0; z < filterList.Count; z++) {
                                var filter = filterList[z];
                                var output = matrixList[z];
                                output[item.Y, item.X] += filter[filterIndex] * error;
                            }
                        }
                    }
                }
            }

            var matrixList2 = matrixList.Select(m => new CpuMatrix(m));
            var ret = new Cpu3DTensor(matrixList2.ToList());
            if (padding > 0)
                return ret.RemovePadding(padding);
            return ret;
        }

        public IMatrix ReverseIm2Col(IReadOnlyList<IReadOnlyList<IVector>> filters, int inputHeight, int inputWidth, int inputDepth, int padding, int filterHeight, int filterWidth, int stride)
        {
            var columns = inputHeight + padding * 2;
            var rows = inputWidth + padding * 2;
            var convolutions = ConvolutionHelper.Default(columns, rows, filterHeight, filterHeight, stride);
            var ret = Enumerable.Range(0, Depth)
                .Select(i => DenseMatrix.Create(columns * rows, inputDepth, 0f))
                .ToList()
            ;
            //var ret2 = DenseMatrix.Create(inputHeight * inputWidth, inputDepth, 0f);

            for (var k = 0; k < Depth; k++) {
                var slice = GetDepthSlice(k).AsIndexable();
                var filterList = filters[k].Select(f => f.AsIndexable()).ToList();
                var output = ret[k];

                foreach (var convolution in convolutions) {
                    var first = convolution.First();
                    var error = slice[first.X / stride, first.Y / stride];
                    //if (error != 0) {
                    foreach (var item in convolution) {
                        var i = item.X - first.X;
                        var j = item.Y - first.Y;
                        var filterIndex = i * filterHeight + j;
                        var outputRow = item.X * inputHeight + item.Y;
                        for (var z = 0; z < inputDepth; z++) {
                            var filter = filterList[z];
                            output[outputRow, z] = filter[filterIndex] * error;
                            //ret2[index2, z] += filter[index] * error;
                        }
                    }
                    //}
                }
            }
            var ret2 = DenseMatrix.Create(inputHeight * inputWidth, inputDepth, 0f);
            foreach(var item in ret)
                ret2 = (DenseMatrix)ret2.Add(item);
            return new CpuMatrix(ret2);
        }
    }
}
