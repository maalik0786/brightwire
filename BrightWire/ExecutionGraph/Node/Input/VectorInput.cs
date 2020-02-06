using System.Collections.Generic;
using System.IO;
using BrightWire.ExecutionGraph.Helper;
using BrightWire.Models;

namespace BrightWire.ExecutionGraph.Node.Input
{
	class VectorInput : NodeBase
	{
		class Backpropagation : BackpropagationBase<VectorInput>
		{
			public Backpropagation(VectorInput source) : base(source) { }

			public override void _Backward(INode fromNode, IGraphData errorSignal, IContext context,
				IReadOnlyList<INode> parents)
			{
				var es = errorSignal.GetMatrix();
				using var columnSums = es.ColumnSums();
				columnSums.Multiply(1f / es.RowCount);

				// store the updates
				var learningContext = context.LearningContext;
				learningContext.StoreUpdate(_source, columnSums, err =>
				{
					var delta = err.AsIndexable();
					for (var j = 0; j < _source.Data.Length; j++)
						_source.Data[j] += delta[j] * context.LearningContext.BatchLearningRate;
				});
			}
		}

		public VectorInput(float[] data, string name = null, string id = null) : base(name, id)
		{
			Data = data;
		}

		public float[] Data { get; }

		public override void ExecuteForward(IContext context)
		{
			var data =
				context.LinearAlgebraProvider.CreateMatrix(context.BatchSequence.MiniBatch.BatchSize,
					Data.Length, (x, y) => Data[y]);
			_AddNextGraphAction(context, new MatrixGraphData(data), () => new Backpropagation(this));
		}

		protected override (string Description, byte[] Data) _GetInfo()
		{
			return ("VI", _WriteData(WriteTo));
		}

		public override void WriteTo(BinaryWriter writer)
		{
			FloatVector.Create(Data).WriteTo(writer);
		}

		public override void ReadFrom(GraphFactory factory, BinaryReader reader)
		{
			FloatVector.ReadFrom(reader).Data.CopyTo(Data, 0);
		}
	}
}