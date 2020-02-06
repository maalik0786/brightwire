using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using BrightWire.ExecutionGraph;
using BrightWire.Models;
using BrightWire.TrainingData.Artificial;
using NUnit.Framework;
using ProtoBuf;

namespace BrightWire.Tests
{
	public class SerializeTests
	{
		public class CustomErrorMetric : IErrorMetric
		{
			public IMatrix CalculateGradient(IContext context, IMatrix output, IMatrix targetOutput) =>
				targetOutput.Subtract(output);

			public float Compute(FloatVector output, FloatVector targetOutput) =>
				output.MaximumIndex() == targetOutput.MaximumIndex() ? 1 : 0;

			public bool DisplayAsPercentage => true;
		}

		static ILinearAlgebraProvider _cpu;
		private static GraphModel bestNetwork;

		static (GraphFactory, IDataSource) MakeGraphAndData()
		{
			var graph = new GraphFactory(_cpu);
			var data = graph.CreateDataSource(And.Get());
			return (graph, data);
		}

		[SetUp]
		public static void Load()
		{
			_cpu = BrightWireProvider.CreateLinearAlgebra(false);
			var (graph, data) = MakeGraphAndData();
			var engine = graph.CreateTrainingEngine(data);
			var errorMetric = new CustomErrorMetric();
			graph.Connect(engine).AddFeedForward(1).Add(graph.SigmoidActivation()).
				AddBackpropagation(errorMetric);
			engine.Train(300, data, errorMetric, bn => bestNetwork = bn);
			AssertEngineGetsGoodResults(engine, data);
		}

		private static void AssertEngineGetsGoodResults(IGraphEngine engine, IDataSource data)
		{
			var results = engine.Execute(data)?.FirstOrDefault();
			Assert.IsNotNull(results);
			bool Handle(FloatVector value) => value.Data[0] > 0.5f;
			Debug.Assert(results.Output.
				Zip(results.Target, (result, target) => Handle(result) == Handle(target)).All(x => x));
		}

		[Test]
		public static void Cleanup() => _cpu.Dispose();

		[Test]
		public void CreateFromExecutionGraph()
		{
			var (graph, data) = MakeGraphAndData();
			var engine = graph.CreateEngine(bestNetwork.Graph);
			AssertEngineGetsGoodResults(engine, data);
		}

		[Test]
		public void DeserialiseExecutionGraph()
		{
			var (graph, data) = MakeGraphAndData();
			Models.ExecutionGraph executionGraphReloaded = null;
			using (var file = new MemoryStream())
			{
				Serializer.Serialize(file, bestNetwork.Graph);
				file.Position = 0;
				executionGraphReloaded = Serializer.Deserialize<Models.ExecutionGraph>(file);
			}

			Assert.IsNotNull(executionGraphReloaded);
			var engine = graph.CreateEngine(executionGraphReloaded);
			AssertEngineGetsGoodResults(engine, data);
		}

		void _AssertEqual<T>(T[] array1, T[] array2)
		{
			Assert.AreEqual(array1.Length, array2.Length);
			for (var i = 0; i < array1.Length; i++)
				Assert.AreEqual(array1[i], array2[i]);
		}

		[Test]
		public void DeserialiseVectorisationModel()
		{
			var builder = BrightWireProvider.CreateDataTableBuilder();
			builder.AddColumn(ColumnType.String, "label");
			builder.AddColumn(ColumnType.String, "output", true);
			builder.Add("a", "0");
			builder.Add("b", "0");
			builder.Add("c", "1");
			var dataTable = builder.Build();
			var vectoriser = dataTable.GetVectoriser();
			var model = vectoriser.GetVectorisationModel();
			var vectorList = new List<FloatVector>();
			dataTable.ForEach(row => vectorList.Add(vectoriser.GetInput(row)));
			var vectoriser2 = dataTable.GetVectoriser(model);
			var vectorList2 = new List<FloatVector>();
			dataTable.ForEach(row => vectorList2.Add(vectoriser2.GetInput(row)));
			foreach (var item in vectorList.Zip(vectorList2, (v1, v2) => (v1, v2)))
				_AssertEqual(item.Item1.Data, item.Item2.Data);
		}
	}
}