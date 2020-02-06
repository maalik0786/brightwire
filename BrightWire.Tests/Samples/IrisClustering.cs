﻿using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;

namespace BrightWire.Tests.Samples
{
	partial class Program
	{
		static void _WriteClusters(IReadOnlyList<IReadOnlyList<IVector>> clusters,
			Dictionary<IVector, string> labelTable)
		{
			foreach (var cluster in clusters)
			{
				foreach (var item in cluster)
					Console.WriteLine(labelTable[item]);
				Console.WriteLine("---------------------------------------------");
			}
		}

		public static void IrisClustering()
		{
			// download the iris data set
			byte[] data;
			using (var client = new WebClient())
			{
				data = client.DownloadData(
					"https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data");
			}

			// parse the iris CSV into a data table
			var dataTable = new StreamReader(new MemoryStream(data)).ParseCSV();

			// the last column is the classification target ("Iris-setosa", "Iris-versicolor", or "Iris-virginica")
			var targetColumnIndex = dataTable.TargetColumnIndex = dataTable.ColumnCount - 1;
			var featureColumns = Enumerable.Range(0, 4).ToList();

			// convert the data table to vectors
			using var lap = BrightWireProvider.CreateLinearAlgebra();
			var rows = dataTable.GetNumericRows(featureColumns).Select(r => lap.CreateVector(r)).ToList();
			var labels = rows.
				Zip(dataTable.GetColumn<string>(targetColumnIndex), (r, l) => Tuple.Create(r, l)).
				ToDictionary(d => d.Item1, d => d.Item2);
			Console.WriteLine("Hierachical Clustering...");
			_WriteClusters(rows.HierachicalCluster(3), labels);
			Console.WriteLine();
			Console.WriteLine("K Means Clustering...");
			_WriteClusters(rows.KMeans(lap, 3), labels);
			Console.WriteLine();
		}
	}
}