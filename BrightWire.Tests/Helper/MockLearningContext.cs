using System;

namespace BrightWire.Tests.Helper
{
	class MockLearningContext : ILearningContext
	{
		public double EpochSeconds { get; set; }
		public long EpochMilliseconds { get; set; }
		public ILinearAlgebraProvider LinearAlgebraProvider { get; set; }
		public int CurrentEpoch { get; set; }
		public float LearningRate { get; set; }
		public float BatchLearningRate { get; set; }
		public int BatchSize { get; set; }
		public int RowCount { get; set; }
		public void StoreUpdate<T>(INode fromNode, T update, Action<T> updater) { }

		public TrainingErrorCalculation TrainingErrorCalculation { get; set; }
		public bool DeferUpdates { get; set; }

		public void ApplyUpdates()
		{
			throw new NotImplementedException();
		}

		public void StartEpoch()
		{
			throw new NotImplementedException();
		}

		public void EndEpoch()
		{
			throw new NotImplementedException();
		}

		public void SetRowCount(int rowCount)
		{
			throw new NotImplementedException();
		}

		public void DeferBackpropagation(IGraphData errorSignal, Action<IGraphData> update)
		{
			throw new NotImplementedException();
		}

		public void BackpropagateThroughTime(IGraphData signal, int maxDepth = 2147483647)
		{
			throw new NotImplementedException();
		}

		public void ScheduleLearningRate(int atEpoch, float newLearningRate)
		{
			throw new NotImplementedException();
		}

		public void EnableNodeUpdates(INode node, bool enableUpdates) =>
			throw new NotImplementedException();

		public void Clear() => throw new NotImplementedException();

		public Action<string> MessageLog { get; set; }
#pragma warning disable CS0067
		public event Action<ILearningContext> BeforeEpochStarts;
		public event Action<ILearningContext> AfterEpochEnds;
		public IErrorMetric ErrorMetric { get; set; }
	}
}