using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using BrightWire.TrainingData.Helper;
using MathNet.Numerics.Distributions;
using NUnit.Framework;

namespace BrightWire.Tests
{
	public class MarkovModelTests
	{
		static string _text;

		[SetUp]
		public static void Load()
		{
			// download some text
			using var client = new WebClient();
			var data = client.DownloadString(new Uri("http://www.gutenberg.net.au/fsf/PAT-HOBBY.txt"));
			var pos = data.IndexOf(@"Project Gutenberg Australia", StringComparison.Ordinal);
			var pos2 = data.IndexOf("THE END", pos, StringComparison.Ordinal);
			_text = data.Substring(pos, pos2 - pos - 1);
		}

		public void _Train(IMarkovModelTrainer<string> trainer)
		{
			var tokens = SimpleTokeniser.Tokenise(_text).ToList();
			var sentences = SimpleTokeniser.FindSentences(tokens).ToList();
			foreach (var sentence in sentences)
				trainer.Add(sentence);
		}

		[Test]
		public void TrainModel2()
		{
			var trainer = BrightWireProvider.CreateMarkovTrainer2<string>();
			_Train(trainer);

			// test serialisation/deserialisation
			using (var buffer = new MemoryStream())
			{
				trainer.SerialiseTo(buffer);
				buffer.Position = 0;
				trainer.DeserialiseFrom(buffer, true);
			}

			var dictionary = trainer.Build().AsDictionary;

			// generate some text
			var rand = new Random();
			string prev = default, curr = default;
			var output = new List<string>();
			for (var i = 0; i < 1024; i++)
			{
				var transitions = dictionary.GetTransitions(prev, curr);
				var distribution =
					new Categorical(transitions.Select(d => Convert.ToDouble(d.Probability)).ToArray());
				var next = transitions[distribution.Sample()].NextState;
				output.Add(next);
				if (SimpleTokeniser.IsEndOfSentence(next))
					break;
				prev = curr;
				curr = next;
			}

			Assert.IsTrue(output.Count < 1024);
		}

		[Test]
		public void TrainModel3()
		{
			var trainer = BrightWireProvider.CreateMarkovTrainer3<string>();
			_Train(trainer);

			// test serialisation/deserialisation
			using (var buffer = new MemoryStream())
			{
				trainer.SerialiseTo(buffer);
				buffer.Position = 0;
				trainer.DeserialiseFrom(buffer, true);
			}

			var dictionary = trainer.Build().AsDictionary;

			// generate some text
			var rand = new Random();
			string prevPrev = default, prev = default, curr = default;
			var output = new List<string>();
			for (var i = 0; i < 1024; i++)
			{
				var transitions = dictionary.GetTransitions(prevPrev, prev, curr);
				var distribution =
					new Categorical(transitions.Select(d => Convert.ToDouble(d.Probability)).ToArray());
				var next = transitions[distribution.Sample()].NextState;
				output.Add(next);
				if (SimpleTokeniser.IsEndOfSentence(next))
					break;
				prevPrev = prev;
				prev = curr;
				curr = next;
			}

			Assert.IsTrue(output.Count < 1024);
		}
	}
}