using BrightWire.TrainingData.Helper;
using NUnit.Framework;

namespace BrightWire.Tests
{
	public class TreeBasedTests
	{
		[Test]
		public void TestDecisionTree()
		{
			var stringTableBuilder = new StringTableBuilder();
			var data = NaiveBayesTests.GetSimpleChineseSet(stringTableBuilder).
				ConvertToWeightedIndexList(false).Vectorise().ConvertToTable(false);
			var model = data.TrainDecisionTree();
			var classifier = model.CreateClassifier();
			var testRows = data.GetRows(new[] { 0, data.RowCount - 1 });
			Assert.IsTrue(classifier.Classify(testRows[0]).GetBestClassification() == "china");
			Assert.IsTrue(classifier.Classify(testRows[1]).GetBestClassification() == "japan");
		}

		[Test]
		public void TestRandomForest()
		{
			var stringTableBuilder = new StringTableBuilder();
			var data = NaiveBayesTests.GetSimpleChineseSet(stringTableBuilder).
				ConvertToWeightedIndexList(false).ConvertToTable();
			var model = data.TrainRandomForest();
			var classifier = model.CreateClassifier();
			var testRows = data.GetRows(new[] { 0, data.RowCount - 1 });
			Assert.IsTrue(classifier.Classify(testRows[0]).GetBestClassification() == "china");
			//Assert.IsTrue(classifier.Classify(testRows[1]).First() == "japan");
		}
	}
}