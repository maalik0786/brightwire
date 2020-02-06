﻿using System;
using System.Collections.Generic;
using System.IO;
using BrightWire.ExecutionGraph;
using BrightWire.ExecutionGraph.Node;
using BrightWire.LinearAlgebra.Helper;

namespace BrightWire.Tests.Samples
{
	partial class Program
	{
		/// <summary>
		/// Example of custom activation function, implemented from:
		/// https://arxiv.org/abs/1706.02515
		/// </summary>
		class SeluActivation : NodeBase
		{
			const float ALPHA = 1.6732632423543772848170429916717f;
			const float SCALE = 1.0507009873554804934193349852946f;

			/// <summary>
			/// Backpropagation of SELU activation
			/// </summary>
			class Backpropagation : SingleBackpropagationBase<SeluActivation>
			{
				public Backpropagation(SeluActivation source) : base(source) { }

				protected override IGraphData _Backpropagate(INode fromNode, IGraphData errorSignal,
					IContext context, IReadOnlyList<INode> parents)
				{
					var matrix = errorSignal.GetMatrix().AsIndexable();
					var delta = context.LinearAlgebraProvider.CreateMatrix(matrix.RowCount,
						matrix.ColumnCount, (i, j) =>
						{
							var x = matrix[i, j];
							if (x >= 0)
								return SCALE;
							return SCALE * ALPHA * BoundMath.Exp(x);
						});
					return errorSignal.ReplaceWith(delta);
				}
			}

			public SeluActivation(string name = null) : base(name) { }

			public override void ExecuteForward(IContext context)
			{
				var matrix = context.Data.GetMatrix().AsIndexable();
				var output = context.LinearAlgebraProvider.CreateMatrix(matrix.RowCount, matrix.ColumnCount,
					(i, j) =>
					{
						var x = matrix[i, j];
						if (x >= 0)
							return SCALE * x;
						return SCALE * (ALPHA * BoundMath.Exp(x) - ALPHA);
					});
				_AddNextGraphAction(context, context.Data.ReplaceWith(output),
					() => new Backpropagation(this));
			}
		}

		public static void TrainWithSelu(string dataFilesPath)
		{
			using var lap = BrightWireProvider.CreateLinearAlgebra();
			var graph = new GraphFactory(lap);

			// parse the iris CSV into a data table and normalise
			var dataTable = new StreamReader(new MemoryStream(File.ReadAllBytes(dataFilesPath))).
				ParseCSV().Normalise(NormalisationType.Standard);

			// split the data table into training and test tables
			var split = dataTable.Split(0);
			var trainingData = graph.CreateDataSource(split.Training);
			var testData = graph.CreateDataSource(split.Test);

			// one hot encoding uses the index of the output vector's maximum value as the classification label
			var errorMetric = graph.ErrorMetric.OneHotEncoding;

			// configure the network properties
			graph.CurrentPropertySet.Use(graph.GradientDescent.RmsProp).Use(
				graph.GaussianWeightInitialisation(true, 0.1f, GaussianVarianceCalibration.SquareRoot2N,
					GaussianVarianceCount.FanInFanOut));

			// create the training engine and schedule a training rate change
			var engine = graph.CreateTrainingEngine(trainingData);
			const int LAYER_SIZE = 64;
			Func<INode> activation = () => new SeluActivation();
			//Func<INode> activation = () => graph.ReluActivation();

			// create the network with the custom activation function
			graph.Connect(engine).AddFeedForward(LAYER_SIZE).AddBatchNormalisation().Add(activation()).
				AddFeedForward(LAYER_SIZE).AddBatchNormalisation().Add(activation()).
				AddFeedForward(LAYER_SIZE).AddBatchNormalisation().Add(activation()).
				AddFeedForward(LAYER_SIZE).AddBatchNormalisation().Add(activation()).
				AddFeedForward(LAYER_SIZE).AddBatchNormalisation().Add(activation()).
				AddFeedForward(LAYER_SIZE).AddBatchNormalisation().Add(activation()).
				AddFeedForward(trainingData.OutputSize).Add(graph.SoftMaxActivation()).
				AddBackpropagation(errorMetric);
			const int TRAINING_ITERATIONS = 500;
			engine.Train(TRAINING_ITERATIONS, testData, errorMetric, null, 50);
		}
	}
}