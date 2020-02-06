using NUnit.Framework;

namespace BrightWire.Tests
{
	public class MiscTests
	{
		//static ILinearAlgebraProvider _lap;

		//[SetUp]
		//public static void Load()
		//{
		//	_lap = BrightWireProvider.CreateLinearAlgebra(false);
		//}

		//[ClassCleanup]
		//public static void Cleanup()
		//{
		//	_lap.Dispose();
		//}

		[Test]
		public void TestFloatConverter()
		{
			var converter = BrightWireProvider.CreateTypeConverter(float.NaN);
			Assert.IsFalse(float.IsNaN((float)converter.ConvertValue("45.5").ConvertedValue));
			Assert.IsTrue(float.IsNaN((float)converter.ConvertValue("sdf").ConvertedValue));
		}
	}
}