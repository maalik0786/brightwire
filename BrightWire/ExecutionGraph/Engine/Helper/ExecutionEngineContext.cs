using System;
using System.Collections.Generic;
using System.Linq;

namespace BrightWire.ExecutionGraph.Engine.Helper
{
	/// <summary>
	/// Execution engine context
	/// </summary>
	class ExecutionEngineContext : IContext
	{
		readonly List<IExecutionHistory> _forward = new List<IExecutionHistory>();
		readonly Dictionary<int, IGraphData> _output = new Dictionary<int, IGraphData>();

		public ExecutionEngineContext(IExecutionContext executionContext, IMiniBatchSequence miniBatch)
		{
			ExecutionContext = executionContext;
			BatchSequence = miniBatch;
			Data = null;
		}

		public void Dispose()
		{
			// nop
		}

		public bool IsTraining => false;
		public INode Source { get; private set; }
		public IExecutionContext ExecutionContext { get; }
		public ILearningContext LearningContext => null;
		public ILinearAlgebraProvider LinearAlgebraProvider => ExecutionContext.LinearAlgebraProvider;
		public IMiniBatchSequence BatchSequence { get; }

		public void AddBackward(IGraphData errorSignal, INode target, INode source) =>
			throw new NotImplementedException();

		public void Backpropagate(IGraphData delta) => throw new NotImplementedException();

		public void AppendErrorSignal(IGraphData errorSignal, INode forNode) =>
			throw new NotImplementedException();

		public void AddForward(IExecutionHistory action, Func<IBackpropagation> callback) =>
			_forward.Add(action);

		public IGraphData ErrorSignal => throw new NotImplementedException();
		public bool HasNext => _forward.Any();
		public IGraphData Data { get; private set; }

		public bool ExecuteNext()
		{
			if (HasNext)
			{
				var next = _forward.ElementAt(0);
				_forward.RemoveAt(0);
				Data = next.Data;
				Source = next.Source;
				if (next.Source.Output != null)
				{
					foreach (var output in next.Source.Output)
						output.SendTo?.ExecuteForward(this, output.Channel);
				}

				return true;
			}

			return false;
		}

		public void SetOutput(IGraphData data, int channel = 0)
		{
			_output[channel] = data;
		}

		public IGraphData GetOutput(int channel = 0)
		{
			if (_output.TryGetValue(channel, out var ret))
				return ret;
			return null;
		}

		public IReadOnlyList<IGraphData> Output =>
			_output.OrderBy(kv => kv.Key).Select(kv => kv.Value).ToList();
	}
}