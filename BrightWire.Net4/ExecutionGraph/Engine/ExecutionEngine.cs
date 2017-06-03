﻿using System.Collections.Generic;
using System.Linq;
using BrightWire.ExecutionGraph.Helper;
using BrightWire.Helper;
using BrightWire.ExecutionGraph.Engine.Helper;
using BrightWire.Models;

namespace BrightWire.ExecutionGraph.Engine
{
    /// <summary>
    /// Executes (without training) graphs
    /// </summary>
    class ExecutionEngine : IGraphEngine
    {
        readonly Models.ExecutionGraph _graph;
        readonly List<(ExecutionEngineContext Context, IMatrix Data)> _executionResults = new List<(ExecutionEngineContext, IMatrix)>();
        readonly ILinearAlgebraProvider _lap;
        IDataSource _dataSource = null;
        readonly INode _input;

        public ExecutionEngine(ILinearAlgebraProvider lap, Models.ExecutionGraph graph, INode input)
        {
            _lap = lap;
            _graph = graph;
            _input = input;
        }

        public Models.ExecutionGraph Graph => _graph;
        public IDataSource DataSource => _dataSource;
        public ILinearAlgebraProvider LinearAlgebraProvider => _lap;
        public INode Input => _input;

        public IReadOnlyList<ExecutionResult> Execute(IDataSource dataSource, int batchSize = 128)
        {
            _lap.PushLayer();
            _dataSource = dataSource;
            var ret = new List<ExecutionResult>();
            var provider = new MiniBatchProvider(dataSource, false);
            using (var executionContext = new ExecutionContext(_lap)) {
                executionContext.Add(provider.GetMiniBatches(batchSize, mb => _Execute(executionContext, mb)));

                IGraphOperation operation;
                while ((operation = executionContext.GetNextOperation()) != null) {
                    _lap.PushLayer();
                    operation.Execute(executionContext);
                    foreach (var item in _executionResults) {
                        ret.Add(new ExecutionResult(item.Context.BatchSequence, item.Data.AsIndexable().Rows.Select(r => r.Data).ToList()));
                        item.Context.Dispose();
                        item.Data?.Dispose();
                    }
                    _executionResults.Clear();
                    _lap.PopLayer();
                }
            }
            _lap.PopLayer();
            _dataSource = null;
            return ret;
        }

        public ExecutionResult Execute(float[] input)
        {
            _lap.PushLayer();
            ExecutionResult ret = null;
            var provider = new MiniBatchProvider(new Helper.SingleRowDataSource(input), false);
            using (var executionContext = new ExecutionContext(_lap)) {
                executionContext.Add(provider.GetMiniBatches(1, mb => _Execute(executionContext, mb)));

                IGraphOperation operation;
                while ((operation = executionContext.GetNextOperation()) != null) {
                    _lap.PushLayer();
                    operation.Execute(executionContext);
                    foreach (var item in _executionResults) {
                        ret = new ExecutionResult(item.Context.BatchSequence, item.Data.AsIndexable().Rows.Select(r => r.Data).ToList());
                        item.Context.Dispose();
                        item.Data?.Dispose();
                    }
                    _executionResults.Clear();
                    _lap.PopLayer();
                }
            }
            _lap.PopLayer();
            _dataSource = null;
            return ret;
        }

        IReadOnlyList<IContext> _Execute(IExecutionContext executionContext, IMiniBatch batch)
        {
            var ret = new List<IContext>();
            if (batch.IsSequential) {
                IMiniBatchSequence curr = null;
                while ((curr = batch.GetNextSequence()) != null) {
                    var context = new ExecutionEngineContext(executionContext, curr);
                    _input.ExecuteForward(context, 0);
                    while (context.HasNext)
                        context.ExecuteNext();
                    _executionResults.Add((context, context.Data.GetMatrix()));
                    ret.Add(context);
                }
            } else {
                var context = new ExecutionEngineContext(executionContext, batch.CurrentSequence);
                _input.ExecuteForward(context, 0);

                while (context.HasNext)
                    context.ExecuteNext();

                _executionResults.Add((context, context.Data.GetMatrix()));
                ret.Add(context);
            }
            return ret;
        }
    }
}