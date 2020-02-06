using System.Linq;
using BrightWire.TrainingData.Helper;
using NUnit.Framework;

namespace BrightWire.Tests
{
	public class ClassificationSetTests
	{
		[Test]
		public void TestTFIDF()
		{
			var stringTableBuilder = new StringTableBuilder();
			var data = NaiveBayesTests.GetSimpleChineseSet(stringTableBuilder);
			Assert.AreEqual(data.Count, 4);
			Assert.AreEqual(data.First().Data.Count, 3);
			var set = data.ConvertToWeightedIndexList(true);
			Assert.AreEqual(set.Count, 2);
			Assert.AreEqual(set.First().Data.Count, 4);
			var tfidf = set.TFIDF();
			Assert.AreEqual(tfidf.Count, 2);
			Assert.AreEqual(tfidf.First().Data.Count, 4);
		}
	}
}