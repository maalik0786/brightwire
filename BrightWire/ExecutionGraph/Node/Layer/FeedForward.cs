using System.Collections.Generic;
using System.IO;
using BrightWire.Models;

namespace BrightWire.ExecutionGraph.Node.Layer
{
	/// <summary>
	/// Feed forward neural network
	/// https://en.wikipedia.org/wiki/Feedforward_neural_network
	/// </summary>
	class FeedForward : NodeBase, IFeedForward
	{
		protected class Backpropagation : SingleBackpropagationBase<FeedForward>
		{
			readonly IMatrix _input;

			public Backpropagation(FeedForward source, IMatrix input) : base(source)
			{
				_input = input;
			}

			protected override void _Dispose(bool isDisposing)
			{
				//_input.Dispose();
			}

			protected override IGraphData _Backpropagate(INode fromNode, IGraphData errorSignal,
				IContext context, IReadOnlyList<INode> parents)
			{
				var es = errorSignal.GetMatrix();

				// work out the next error signal
				var ret = es.TransposeAndMultiply(_source.Weight);

				// calculate the update to the weights
				var weightUpdate = _input.TransposeThisAndMultiply(es);

				// store the updates
				var learningContext = context.LearningContext;
				learningContext.StoreUpdate(_source, es, err => _source.UpdateBias(err, learningContext));
				learningContext.StoreUpdate(_source, weightUpdate,
					err => _source.UpdateWeights(err, learningContext));
				return errorSignal.ReplaceWith(ret);
			}
		}

		IGradientDescentOptimisation _updater;

		public FeedForward(int inputSize, int outputSize, IVector bias, IMatrix weight,
			IGradientDescentOptimisation updater, string name = null) : base(name)
		{
			Bias = bias;
			Weight = weight;
			_updater = updater;
			InputSize = inputSize;
			OutputSize = outputSize;
		}

		public IVector Bias { get; private set; }
		public IMatrix Weight { get; private set; }
		public int InputSize { get; private set; }
		public int OutputSize { get; private set; }

		protected override void _Dispose(bool isDisposing)
		{
			Bias.Dispose();
			Weight.Dispose();
		}

		public void UpdateWeights(IMatrix delta, ILearningContext context)
		{
			_updater.Update(Weight, delta, context);
		}

		public void UpdateBias(IMatrix delta, ILearningContext context)
		{
			using var columnSums = delta.ColumnSums();
			Bias.AddInPlace(columnSums, 1f / delta.RowCount, context.BatchLearningRate);
		}

		protected IMatrix _FeedForward(IMatrix input, IMatrix weight)
		{
			var output = input.Multiply(weight);
			output.AddToEachRow(Bias);
			return output;
		}

		public override void ExecuteForward(IContext context)
		{
			var input = context.Data.GetMatrix();
			var output = _FeedForward(input, Weight);

			// set output
			_AddNextGraphAction(context, context.Data.ReplaceWith(output),
				() => new Backpropagation(this, input));
		}

		protected override (string Description, byte[] Data) _GetInfo()
		{
			return ("FF", _WriteData(WriteTo));
		}

		public override void ReadFrom(GraphFactory factory, BinaryReader reader)
		{
			var lap = factory?.LinearAlgebraProvider;
			InputSize = reader.ReadInt32();
			OutputSize = reader.ReadInt32();

			// read the bias parameters
			var bias = FloatVector.ReadFrom(reader);
			if (Bias == null)
				Bias = lap.CreateVector(bias);
			else
				Bias.Data = bias;

			// read the weight parameters
			var weight = FloatMatrix.ReadFrom(reader);
			if (Weight == null)
				Weight = lap.CreateMatrix(weight);
			else
				Weight.Data = weight;
			if (_updater == null)
				_updater = factory?.CreateWeightUpdater(Weight);
		}

		public override void WriteTo(BinaryWriter writer)
		{
			writer.Write(InputSize);
			writer.Write(OutputSize);
			Bias.Data.WriteTo(writer);
			Weight.Data.WriteTo(writer);
		}
	}
}