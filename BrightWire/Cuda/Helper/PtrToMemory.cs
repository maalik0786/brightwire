using System.Threading;
using ManagedCuda;
using ManagedCuda.BasicTypes;

namespace BrightWire.Cuda.Helper
{
	/// <summary>
	/// A pointer to a block of device memory (the block is owned by another pointer)
	/// </summary>
	class PtrToMemory : IDeviceMemoryPtr
	{
		readonly IDeviceMemoryPtr _rootBlock;
		readonly CudaContext _context;
		int refCount = 1;

		public PtrToMemory(CudaContext context, IDeviceMemoryPtr rootBlock, CUdeviceptr ptr, SizeT size)
		{
			_context = context;
			DeviceVariable = new CudaDeviceVariable<float>(ptr, size);
			_rootBlock = rootBlock;
			rootBlock.AddRef();
		}

		public CudaDeviceVariable<float> DeviceVariable { get; }
		public CUdeviceptr DevicePointer => DeviceVariable.DevicePointer;
		public int Size => DeviceVariable.Size;

		public void Clear()
		{
			_context.ClearMemory(DeviceVariable.DevicePointer, 0, DeviceVariable.SizeInBytes);
		}

		public void CopyToDevice(float[] source)
		{
			DeviceVariable.CopyToDevice(source);
		}

		public void CopyToDevice(IDeviceMemoryPtr source)
		{
			DeviceVariable.CopyToDevice(source.DeviceVariable);
		}

		public void CopyToHost(float[] target)
		{
			_context.CopyToHost<float>(target, DeviceVariable.DevicePointer);
		}

		public int AddRef()
		{
			return Interlocked.Increment(ref refCount) + _rootBlock.AddRef();
		}

		public void Free()
		{
			_rootBlock.Free();
			if (Interlocked.Decrement(ref refCount) <= 0)
				DeviceVariable.Dispose();
		}
	}
}