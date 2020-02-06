using System.Collections.Generic;
using System.Linq;

namespace BrightWire.ExecutionGraph.DataTableAdaptor
{
	/// <summary>
	/// Vectorises each row of the data table on demand
	/// </summary>
	class DefaultDataTableAdaptor : RowBasedDataTableAdaptorBase, IRowEncoder,
		IHaveDataTableVectoriser
	{
		public DefaultDataTableAdaptor(ILinearAlgebraProvider lap, IDataTable dataTable,
			IDataTableVectoriser vectoriser = null) : base(lap, dataTable)
		{
			Vectoriser = vectoriser ?? dataTable.GetVectoriser();
		}

		public override IDataSource CloneWith(IDataTable dataTable)
		{
			return new DefaultDataTableAdaptor(_lap, dataTable, Vectoriser);
		}

		public override int InputSize => Vectoriser.InputSize;
		public override int OutputSize => Vectoriser.OutputSize;
		public override bool IsSequential => false;
		public IDataTableVectoriser Vectoriser { get; }

		public float[] Encode(IRow row)
		{
			return Vectoriser.GetInput(row).Data;
		}

		public override IMiniBatch Get(IExecutionContext executionContext, IReadOnlyList<int> rows)
		{
			var data = _GetRows(rows).Select(r => (new[] { Encode(r) }, Vectoriser.GetOutput(r).Data)).
				ToList();
			return _GetMiniBatch(rows, data);
		}
	}
}