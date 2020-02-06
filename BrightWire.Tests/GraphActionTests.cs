using BrightWire.ExecutionGraph;
using BrightWire.Tests.Helper;
using NUnit.Framework;

namespace BrightWire.Tests
{
	public class GraphActionTests
	{
		static ILinearAlgebraProvider _cpu;
		static GraphFactory _factory;

		[SetUp]
		public static void Load()
		{
			_cpu = BrightWireProvider.CreateLinearAlgebra(false);
			_factory = new GraphFactory(_cpu);
		}

		[Test]
		public static void Cleanup() => _cpu.Dispose();

		public void _TestAction(IAction action, IGraphData input, IGraphData expectedOutput)
		{
			var context = new TestingContext(_cpu);
			var output = action.Execute(input, context);
			FloatingPointHelper.AssertEqual(output.GetMatrix().AsIndexable(),
				expectedOutput.GetMatrix().AsIndexable());
		}

		[Test]
		public void TestConstrainInput()
		{
			var input = _cpu.CreateVector(new[] { -1.5f, -1f, -0.5f, 0, 0.5f, 1f, 1.5f }).
				ReshapeAsMatrix(1, 7);
			var output = _cpu.CreateVector(new[] { -1f, -1f, -0.5f, 0, 0.5f, 1f, 1f }).
				ReshapeAsMatrix(1, 7);
			_TestAction(_factory.GraphAction.Constrain(), input.AsGraphData(), output.AsGraphData());
		}
	}
}