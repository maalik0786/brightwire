using System.Collections.Generic;
using System.Linq;

namespace BrightWire.TabularData.Analysis
{
	/// <summary>
	/// Collects min and max lengths and the set of distinct items from a single string column of a data table
	/// </summary>
	internal class StringCollector : IRowProcessor, IStringColumnInfo
	{
		readonly int _maxDistinct;
		readonly Dictionary<string, ulong> _distinct = new Dictionary<string, ulong>();

		ulong _highestCount;
		string _mode;

		public StringCollector(int index, int maxDistinct = 131072 * 4)
		{
			ColumnIndex = index;
			_maxDistinct = maxDistinct;
		}

		public bool Process(IRow row)
		{
			var val = row.GetField<string>(ColumnIndex);
			var len = val.Length;
			if (len < MinLength)
				MinLength = len;
			if (len > MaxLength)
				MaxLength = len;

			// add to distinct values
			if (_distinct.Count < _maxDistinct)
			{
				ulong count = 0;
				if (_distinct.TryGetValue(val, out var temp))
					_distinct[val] = count = temp + 1;
				else
					_distinct.Add(val, count = 1);
				if (count > _highestCount)
				{
					_highestCount = count;
					_mode = val;
				}
			}

			return true;
		}

		public int ColumnIndex { get; }
		public int MinLength { get; private set; } = int.MaxValue;
		public int MaxLength { get; private set; } = int.MinValue;
		public string MostCommonString => _distinct.Count < _maxDistinct ? _mode : null;
		public int? NumDistinct => _distinct.Count < _maxDistinct ? _distinct.Count : (int?)null;
		public IEnumerable<object> DistinctValues
		{
			get { return _distinct.Count < _maxDistinct ? _distinct.Select(kv => kv.Key) : null; }
		}
		public ColumnInfoType Type => ColumnInfoType.String;
	}
}