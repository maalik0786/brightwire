﻿using BrightWire.Connectionist.Training.Helper;
using BrightWire.Helper;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using BrightWire.Models;

namespace BrightWire.Connectionist.Training.Batch
{
    public class BatchTrainer : INeuralNetworkTrainer
    {
        readonly bool _stochastic, _calculateTrainingError;
        readonly int _inputSize, _outputSize;
        readonly IReadOnlyList<INeuralNetworkLayerTrainer> _layer;

        const int DEFAULT_BATCH_SIZE = 128;

        public BatchTrainer(IReadOnlyList<INeuralNetworkLayerTrainer> layer, bool stochastic, bool calculateTrainingError)
        {
            _layer = layer;
            _stochastic = stochastic;
            _calculateTrainingError = calculateTrainingError;
            _inputSize = layer.First().LayerUpdater.Layer.InputSize;
            _outputSize = layer.Last().LayerUpdater.Layer.OutputSize;
        }

        protected virtual void Dispose(bool disposing)
        {
            if(disposing) {
                foreach (var item in _layer)
                    item.Dispose();
            }
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        public IReadOnlyList<INeuralNetworkLayerTrainer> Layer { get { return _layer; } }

        public FeedForwardNetwork NetworkInfo
        {
            get
            {
                return new FeedForwardNetwork {
                    Layer = _layer.Select(l => l.LayerUpdater.Layer.LayerInfo).ToArray()
                };
            }

            set
            {
                if (value.Layer != null) {
                    for (int i = 0, len = value.Layer.Length, len2 = _layer.Count; i < len && i < len2; i++)
                        _layer[i].LayerUpdater.Layer.LayerInfo = value.Layer[i];
                }
            }
        }

        public ITrainingContext CreateContext(float trainingRate, int batchSize)
        {
            return new TrainingContext(trainingRate, batchSize);
        }

        IEnumerable<Tuple<IMatrix, IMatrix>> _GetMiniBatches(ITrainingDataProvider data, bool shuffle, int batchSize)
        {
            // shuffle the training data order
            var range = Enumerable.Range(0, data.Count);
            var iterationOrder = (shuffle ? range.Shuffle() : range).ToList();

            for (var j = 0; j < data.Count; j += batchSize) {
                var maxRows = Math.Min(iterationOrder.Count, batchSize + j) - j;
                var rows = iterationOrder.Skip(j).Take(maxRows).ToList();
                yield return data.GetTrainingData(rows, _inputSize, _outputSize);
            }
        }

        public void Train(ITrainingDataProvider trainingData, int numEpochs, ITrainingContext context)
        {
            IMatrix curr = null;
            for (int i = 0; i < numEpochs && context.ShouldContinue; i++) {
                context.StartEpoch(trainingData.Count);
                var batchErrorList = new List<double>();

                // iterate over each mini batch
                foreach (var miniBatch in _GetMiniBatches(trainingData, _stochastic, context.MiniBatchSize)) {
                    var garbage = new List<IMatrix>();
                    garbage.Add(curr = miniBatch.Item1);

                    // feed forward
                    var layerStack = new Stack<INeuralNetworkLayerTrainer>();
                    foreach (var layer in _layer) {
                        garbage.Add(curr = layer.FeedForward(curr, true));
                        layerStack.Push(layer);
                    }

                    // calculate the error against the training examples
                    using (var expectedOutput = miniBatch.Item2) {
                        garbage.Add(curr = expectedOutput.Subtract(curr));

                        // calculate the training error for this mini batch
                        if(_calculateTrainingError)
                            batchErrorList.Add(curr.AsIndexable().Values.Select(v => Math.Pow(v, 2)).Average() / 2);

                        // backpropagate the error
                        while (layerStack.Any()) {
                            var currentLayer = layerStack.Pop();
                            garbage.Add(curr = currentLayer.Backpropagate(curr, context, layerStack.Any()));
                        }
                    }

                    // clear memory
                    garbage.ForEach(m => m?.Dispose());
                    context.EndBatch();
                }
                context.EndEpoch(_calculateTrainingError ? batchErrorList.Average() : 0f);
            }
        }

        public float CalculateCost(ITrainingDataProvider data, ICostFunction costFunction)
        {
            return Execute(data).Select(r => costFunction.Calculate(r.Item1, r.Item2)).Average();
        }

        public IEnumerable<IIndexableVector[]> ExecuteToLayer(ITrainingDataProvider data, int layerDepth)
        {
            IMatrix curr = null;
            foreach (var miniBatch in _GetMiniBatches(data, false, DEFAULT_BATCH_SIZE)) {
                var garbage = new List<IMatrix>();
                garbage.Add(curr = miniBatch.Item1);
                garbage.Add(miniBatch.Item2);

                // feed forward
                for(var i = 0; i < layerDepth; i++) {
                    var layer = _layer[i];
                    garbage.Add(curr = layer.FeedForward(curr, false));
                }

                var ret = curr.AsIndexable().Rows.ToList();

                // clear memory
                garbage.ForEach(m => m.Dispose());

                yield return ret.ToArray();
            }
        }

        public IReadOnlyList<Tuple<IIndexableVector, IIndexableVector>> Execute(ITrainingDataProvider data)
        {
            IMatrix curr = null;
            var ret = new List<Tuple<IIndexableVector, IIndexableVector>>();

            foreach (var miniBatch in _GetMiniBatches(data, false, DEFAULT_BATCH_SIZE)) {
                var garbage = new List<IMatrix>();
                garbage.Add(curr = miniBatch.Item1);
                garbage.Add(miniBatch.Item2);

                // feed forward
                foreach (var layer in _layer)
                    garbage.Add(curr = layer.FeedForward(curr, false));

                // break the output into rows
                ret.AddRange(curr.AsIndexable().Rows.Zip(miniBatch.Item2.AsIndexable().Rows, (a, e) => Tuple.Create(a, e)));

                // clear memory
                garbage.ForEach(m => m.Dispose());
            }
            return ret;
        }
    }
}