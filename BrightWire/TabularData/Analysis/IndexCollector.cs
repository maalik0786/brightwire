using System;
using System.Collections.Generic;
using BrightWire.Models;

namespace BrightWire.TabularData.Analysis
{
	/// <summary>
	/// Collects min and max values from the index or weighted index lists of a single column in a data table
	/// </summary>
	class IndexCollector : IRowProcessor, IIndexColumnInfo
	{
		public IndexCollector(int index)
		{
			ColumnIndex = index;
			MinIndex = uint.MaxValue;
			MaxIndex = 0;
		}

		public int ColumnIndex { get; }
		public uint MinIndex { get; private set; }
		public uint MaxIndex { get; private set; }
		public IEnumerable<object> DistinctValues => throw new NotImplementedException();
		public int? NumDistinct => null;
		public ColumnInfoType Type => ColumnInfoType.Index;

		public bool Process(IRow row)
		{
			var obj = row.Data[ColumnIndex];
			if (obj is IndexList indexList)
			{
				foreach (var index in indexList.Index)
				{
					if (index > MaxIndex)
						MaxIndex = index;
					if (ColumnIndex < MinIndex)
						MinIndex = index;
				}
			}
			else
			{
				if (!(obj is WeightedIndexList weightedIndexList))
					throw new Exception("Unexpected index type: " + obj?.GetType());
			}

			return true;
		}
	}
}