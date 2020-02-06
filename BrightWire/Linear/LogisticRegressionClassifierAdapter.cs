using System.Collections.Generic;

namespace BrightWire.Linear
{
	class LogisticRegressionClassifierAdapter : IRowClassifier
	{
		readonly ILogisticRegressionClassifier _classifier;
		readonly IReadOnlyList<int> _attributeColumns;
		readonly string _positiveLabel;
		readonly string _negativeLabel;

		public LogisticRegressionClassifierAdapter(ILogisticRegressionClassifier classifier,
			IReadOnlyList<int> attributeColumns, string negativeLabel = "0", string positiveLabel = "1")
		{
			_classifier = classifier;
			_positiveLabel = positiveLabel;
			_negativeLabel = negativeLabel;
			_attributeColumns = attributeColumns;
		}

		public IReadOnlyList<(string Label, float Weight)> Classify(IRow row)
		{
			var prediction = _classifier.Predict(row.GetFields<float>(_attributeColumns));
			return new[] { (prediction >= 0.5f ? _positiveLabel : _negativeLabel, 1f) };
		}
	}
}