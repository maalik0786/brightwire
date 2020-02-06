using System;
using System.Linq;
using BrightWire.ExecutionGraph;
using BrightWire.Models;
using BrightWire.TrainingData.Artificial;
using NUnit.Framework;

namespace BrightWire.Tests
{
	public class ConnectionistTests
	{
		static ILinearAlgebraProvider _lap;

		[SetUp]
		public static void Load() => _lap = BrightWireProvider.CreateLinearAlgebra(false);

		[Test]
		public static void Cleanup() => _lap.Dispose();

		[Test]
		public void TiedAutoEncoder()
		{
			const int DATA_SIZE = 1000, REDUCED_SIZE = 200;

			// create some random data
			var rand = new Random();
			var builder = BrightWireProvider.CreateDataTableBuilder();
			builder.AddVectorColumn(DATA_SIZE, "Input");
			builder.AddVectorColumn(DATA_SIZE, "Output", true);
			for (var i = 0; i < 100; i++)
			{
				var vector = new FloatVector
				{
					Data = Enumerable.Range(0, DATA_SIZE).Select(j => Convert.ToSingle(rand.NextDouble())).
						ToArray()
				};
				builder.Add(vector, vector);
			}

			var dataTable = builder.Build();

			// build the autoencoder with tied weights
			var graph = new GraphFactory(_lap);
			var dataSource = graph.CreateDataSource(dataTable);
			var engine = graph.CreateTrainingEngine(dataSource, 0.03f, 32);
			var errorMetric = graph.ErrorMetric.Quadratic;
			graph.CurrentPropertySet.Use(graph.RmsProp()).Use(graph.WeightInitialisation.Xavier);
			graph.Connect(engine).AddFeedForward(REDUCED_SIZE, "layer").Add(graph.TanhActivation()).
				AddTiedFeedForward(engine.Start.FindByName("layer") as IFeedForward).
				Add(graph.TanhActivation()).AddBackpropagation(errorMetric);
			using (var executionContext = graph.CreateExecutionContext())
			{
				for (var i = 0; i < 2; i++)
				{
					var trainingError = engine.Train(executionContext);
				}
			}

			var networkGraph = engine.Graph;
			var executionEngine = graph.CreateEngine(networkGraph);
			var results = executionEngine.Execute(dataTable.GetRow(0).GetField<FloatVector>(0).Data);
		}

		[Test]
		public void TestRecurrent()
		{
			var data = BinaryIntegers.Addition(100, false).Split(0);
			var graph = new GraphFactory(_lap);
			var errorMetric = graph.ErrorMetric.BinaryClassification;
			graph.CurrentPropertySet.Use(graph.GradientDescent.Adam).Use(
				graph.GaussianWeightInitialisation(false, 0.1f, GaussianVarianceCalibration.SquareRoot2N));

			// create the engine
			var trainingData = graph.CreateDataSource(data.Training);
			var testData = trainingData.CloneWith(data.Test);
			var engine = graph.CreateTrainingEngine(trainingData, learningRate: 0.01f, batchSize: 16);

			// build the network
			const int HIDDEN_LAYER_SIZE = 32, TRAINING_ITERATIONS = 5;
			var memory = new float[HIDDEN_LAYER_SIZE];
			var network = graph.Connect(engine).AddSimpleRecurrent(graph.ReluActivation(), memory).
				AddFeedForward(engine.DataSource.OutputSize).Add(graph.ReluActivation()).
				AddBackpropagationThroughTime(errorMetric);

			// train the network for twenty iterations, saving the model on each improvement
			Models.ExecutionGraph bestGraph = null;
			engine.Train(TRAINING_ITERATIONS, testData, errorMetric, bn => bestGraph = bn.Graph);

			// export the graph and verify it against some unseen integers on the best model
			var executionEngine = graph.CreateEngine(bestGraph ?? engine.Graph);
			var testData2 = graph.CreateDataSource(BinaryIntegers.Addition(8, true));
			var results = executionEngine.Execute(testData2);
		}
	}
}