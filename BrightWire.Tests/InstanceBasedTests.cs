using System.Linq;
using NUnit.Framework;

namespace BrightWire.Tests
{
	public class InstanceBasedTests
	{
		static ILinearAlgebraProvider _lap;

		[SetUp]
		public static void Load() => _lap = BrightWireProvider.CreateLinearAlgebra(false);

		[Test]
		public static void Cleanup() => _lap.Dispose();

		[Test]
		public void KNN()
		{
			var dataTable = BrightWireProvider.CreateDataTableBuilder();
			dataTable.AddColumn(ColumnType.Float, "height");
			dataTable.AddColumn(ColumnType.Int, "weight").IsContinuous = true;
			dataTable.AddColumn(ColumnType.Int, "foot-size").IsContinuous = true;
			dataTable.AddColumn(ColumnType.String, "gender", true);

			// sample data from: https://en.wikipedia.org/wiki/Naive_Bayes_classifier
			dataTable.Add(6f, 180, 12, "male");
			dataTable.Add(5.92f, 190, 11, "male");
			dataTable.Add(5.58f, 170, 12, "male");
			dataTable.Add(5.92f, 165, 10, "male");
			dataTable.Add(5f, 100, 6, "female");
			dataTable.Add(5.5f, 150, 8, "female");
			dataTable.Add(5.42f, 130, 7, "female");
			dataTable.Add(5.75f, 150, 9, "female");
			var index = dataTable.Build();
			var testData = BrightWireProvider.CreateDataTableBuilder(dataTable.Columns);
			var row = testData.Add(6f, 130, 8, "?");
			var model = index.TrainKNearestNeighbours();
			var classifier = model.CreateClassifier(_lap, 2);
			var classification = classifier.Classify(row);
			Assert.IsTrue(classification.OrderByDescending(c => c.Weight).First().Label == "female");
		}
	}
}