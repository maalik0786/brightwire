﻿using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Text;
using System.Threading.Tasks;

namespace BrightWire.SampleCode
{
    partial class Program 
    {
        public static void MultinomialLogisticRegression()
        {
            // download the iris data set
            byte[] data;
            using (var client = new WebClient()) {
                data = client.DownloadData("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data");
            }

            // parse the iris CSV into a data table
            var dataTable = new StreamReader(new MemoryStream(data)).ParseCSV(',');

            // the last column is the classification target ("Iris-setosa", "Iris-versicolor", or "Iris-virginica")
            var targetColumnIndex = dataTable.TargetColumnIndex = dataTable.ColumnCount - 1;

            // split the data table into training and test tables
            var split = dataTable.Split(0);

            // create three training data tables in which the class label is replaced by a boolean corresponding to true for each respective class label
            var trainingData = dataTable.Analysis[targetColumnIndex].DistinctValues
                .Cast<string>()
                .Select(cls => Tuple.Create(cls, split.Item1.Project(r => {
                    var row = r.Take(4).ToList();
                    row.Add((string)r[targetColumnIndex] == cls);
                    return row;
                })))
                .ToList()
            ;
            using (var lap = Provider.CreateCPULinearAlgebra()) {
                // train three classifiers on each training data set
                var classifiers = trainingData.Select(t =>
                    Tuple.Create(t.Item1, t.Item2.TrainLogisticRegression(lap, 500, 0.1f).CreatePredictor(lap))
                ).ToList();

                // evaluate each classifier on each row of the test data and pick the highest scoring classification
                var testData = split.Item2.GetNumericRows(Enumerable.Range(0, 4));
                var testLabels = split.Item2.GetColumn<string>(targetColumnIndex);
                var predictions = testData.Select(row => classifiers
                    .Select(m => Tuple.Create(m.Item1, m.Item2.Predict(row)))
                    .OrderByDescending(r => r.Item2)
                    .Select(r => r.Item1)
                    .First()
                );
                Console.WriteLine("Multinomial logistic regression accuracy: {0:P}", predictions.Zip(testLabels, (p, a) => p == a ? 1 : 0).Average());
            }
        }
    }
}