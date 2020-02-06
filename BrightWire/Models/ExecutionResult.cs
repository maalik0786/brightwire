using System.Collections.Generic;
using System.Linq;

namespace BrightWire.Models
{
	/// <summary>
	/// The output from a mini batch
	/// </summary>
	public class ExecutionResult
	{
		/// <summary>
		/// Constructor
		/// </summary>
		/// <param name="miniBatch">The mini batch sequence</param>
		/// <param name="output">The mini batch output</param>
		/// <param name="index">Output index</param>
		public ExecutionResult(IMiniBatchSequence miniBatch, IReadOnlyList<FloatVector> output,
			int index)
		{
			Index = index;
			MiniBatchSequence = miniBatch;
			Output = output;
			Target = MiniBatchSequence.Target?.GetMatrix().Data.Row;
			Input = MiniBatchSequence.Input.Select(input => input.GetMatrix().Data.Row).ToList();
		}

		/// <summary>
		/// Output index
		/// </summary>
		public int Index { get; }

		/// <summary>
		/// The list of output rows
		/// </summary>
		public IReadOnlyList<FloatVector> Output { get; }

		/// <summary>
		/// The list of target rows
		/// </summary>
		public IReadOnlyList<FloatVector> Target { get; }

		/// <summary>
		/// The list of input rows
		/// </summary>
		public IReadOnlyList<IReadOnlyList<FloatVector>> Input { get; }

		/// <summary>
		/// The mini batch
		/// </summary>
		public IMiniBatchSequence MiniBatchSequence { get; }

		/// <summary>
		/// Calculates the error of the output against the target
		/// </summary>
		/// <param name="errorMetric">The error metric to calculate with</param>
		/// <returns></returns>
		public float CalculateError(IErrorMetric errorMetric) =>
			Output.Zip(Target, (o, t) => errorMetric.Compute(o, t)).Average();
	}
}