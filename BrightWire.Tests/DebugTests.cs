using BrightWire.Cuda;
using NUnit.Framework;

namespace BrightWire.Tests
{
	/// <summary>
	/// These tests use information that is only set during Debug builds
	/// </summary>
	public class DebugTests
	{
		[Test]
		public void MemoryLayerTest()
		{
#if DEBUG
			using var context = BrightWireGpuProvider.CreateLinearAlgebra();
			var matrix = context.CreateZeroMatrix(10, 10);
			context.PushLayer();
			var matrix2 = context.CreateZeroMatrix(10, 10);
			context.PopLayer();
			Assert.IsFalse(matrix2.IsValid);
			Assert.IsTrue(matrix.IsValid);
			matrix.Dispose();
			Assert.IsFalse(matrix.IsValid);
#endif
		}
	}
}