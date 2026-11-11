using System.Diagnostics;
using System.Globalization;
using RecursiveAI.OneGpu;
using RecursiveAI.OneGpu.Kernels;

// Memory-bound throughput of the large element-wise, reduction, softmax, normalisation and transpose cases. GPU time comes from timestamp
// queries around the kernel's pass; wall time is the host clock around submit and wait. Each row is the median of the timed runs, and the
// bandwidth is the bytes the kernel must move once (reads plus writes of the tensors) over the GPU time.
//
//   dotnet run -c Release --project bench/RecursiveAI.OneGpu.Kernels.Bench [-- <filter>]

var filter = args.Length > 0 ? args[0] : "";
using var device = GpuBackends.CreateDevice(new GpuDeviceOptions { ApplicationName = "onegpu-kernels-bench" });
Console.WriteLine($"# {device.Adapter.Name} | {device.Backend} | timestamps: {device.Capabilities.Timestamps}");

const int Rows = 32768, Cols = 1024, N = Rows * Cols;   // 32 Mi elements: 128 MiB as float32
var data = new float[N];
var rng = 12345u;
for (var i = 0; i < N; i++)
{
    rng = (rng * 1664525u) + 1013904223u;
    data[i] = ((rng >> 8) / 16777216f * 8f) - 4f;
}

var halves = new Half[N];
for (var i = 0; i < N; i++)
{
    halves[i] = (Half)data[i];
}

using var f32a = device.CreateBuffer<float>(data);
using var f32b = device.CreateBuffer<float>(data);
using var f32o = device.CreateBuffer<float>(N);
using var f32s = device.CreateBuffer<float>(Rows);
using var f16a = device.CreateBuffer<Half>(halves);
using var f16b = device.CreateBuffer<Half>(halves);
using var f16o = device.CreateBuffer<Half>(N);
using var scale = device.CreateBuffer<float>(Cols);
using var elementwise = new ElementwiseKernels(device);
using var reduce = new ReduceKernels(device);
using var softmax = new SoftmaxKernels(device);
using var norm = new NormKernels(device);
using var transpose = new TransposeKernels(device);
Console.WriteLine($"# variants: reduce {reduce.Variant}, softmax {softmax.Variant}, norm {norm.Variant}");
Console.WriteLine("case | gpu ms | wall ms | GB/s (gpu) | bytes moved");

const long F32 = sizeof(float), F16 = sizeof(ushort);
Run("add f32 [32Mi]", 3 * N * F32, e => elementwise.EncodeBinary(e, GpuBinaryOp.Add, f32a, 0, [N], f32b, 0, [N], f32o, 0));
Run("add f16 [32Mi]", 3 * N * F16, e => elementwise.EncodeBinary(e, GpuBinaryOp.Add, f16a, 0, [N], f16b, 0, [N], f16o, 0));
Run("bias add + gelu f32 [32768,1024]", (2 * N + Cols) * F32, e => elementwise.EncodeBinary(e, GpuBinaryOp.Add, f32a, 0, [Rows, Cols], scale, 0, [Cols], f32o, 0, new GpuActivation(GpuUnaryOp.GeluErf)));
Run("gelu f32 [32Mi]", 2 * N * F32, e => elementwise.EncodeUnary(e, GpuUnaryOp.GeluTanh, f32a, 0, f32o, 0, N));
Run("exp f32 [32Mi]", 2 * N * F32, e => elementwise.EncodeUnary(e, GpuUnaryOp.Exp, f32a, 0, f32o, 0, N));
Run("sum over last axis f32 fast [32768,1024]", N * F32, e => reduce.EncodeReduce(e, GpuReduceOp.Sum, f32a, 0, [Rows, Cols], [1], f32s, 0));
Run("sum over last axis f32 deterministic [32768,1024]", N * F32, e => reduce.EncodeReduce(e, GpuReduceOp.Sum, f32a, 0, [Rows, Cols], [1], f32s, 0, GpuKernelMode.Deterministic));
Run("sum over first axis f32 fast [32768,1024]", N * F32, e => reduce.EncodeReduce(e, GpuReduceOp.Sum, f32a, 0, [Rows, Cols], [0], scale, 0));
Run("sum over all f32 fast [32Mi]", N * F32, e => reduce.EncodeReduce(e, GpuReduceOp.Sum, f32a, 0, [N], [0], f32s, 0));
Run("softmax f32 fast [32768,1024]", 2 * N * F32, e => softmax.Encode(e, f32a, 0, [Rows, Cols], 1, f32o, 0));
Run("softmax f32 deterministic [32768,1024]", 2 * N * F32, e => softmax.Encode(e, f32a, 0, [Rows, Cols], 1, f32o, 0, mode: GpuKernelMode.Deterministic));
Run("softmax f16 fast [32768,1024]", 2 * N * F16, e => softmax.Encode(e, f16a, 0, [Rows, Cols], 1, f16o, 0));
Run("softmax f32 fast [524288,64] (short rows)", 2 * N * F32, e => softmax.Encode(e, f32a, 0, [N / 64, 64], 1, f32o, 0));
Run("layer norm f32 fast [32768,1024]", 2 * N * F32, e => norm.Encode(e, false, f32a, 0, Rows, Cols, scale, 0, scale, 0, f32o, 0));
Run("layer norm f32 deterministic [32768,1024]", 2 * N * F32, e => norm.Encode(e, false, f32a, 0, Rows, Cols, scale, 0, scale, 0, f32o, 0, mode: GpuKernelMode.Deterministic));
Run("rms norm f16 fast [32768,1024]", 2 * N * F16, e => norm.Encode(e, true, f16a, 0, Rows, Cols, null, 0, null, 0, f16o, 0));
Run("transpose f32 [8192,4096]", 2 * N * F32, e => transpose.Encode(e, f32a, 0, [8192, 4096], [1, 0], f32o, 0));
Run("split heads f32 [32,1024,12,64 -> 32,12,1024,64]", 2 * 32L * 1024 * 768 * F32, e => transpose.EncodeSplitHeads(e, f32a, 0, 32, 1024, 12, 64, f32o, 0));

void Run(string name, long bytes, Action<GpuCommandEncoder> record)
{
    if (filter.Length > 0 && !name.Contains(filter, StringComparison.Ordinal))
    {
        return;
    }

    using var stamps = device.Capabilities.Timestamps ? device.CreateTimestampSet(2) : null;
    var gpu = new List<double>();
    var wall = new List<double>();
    Span<ulong> t = stackalloc ulong[2];
    for (var i = 0; i < 25; i++)
    {
        var enc = device.BeginCommands();
        if (stamps is not null)
        {
            enc.BeginPassTiming(new PassTiming(stamps, 0, 1));
        }

        record(enc);
        if (stamps is not null)
        {
            enc.EndPassTiming();
        }

        var sw = Stopwatch.StartNew();
        device.Submit(enc);
        device.WaitIdle();
        sw.Stop();
        if (i < 5)
        {
            continue;
        }

        wall.Add(sw.Elapsed.TotalMilliseconds);
        if (stamps is not null && device.TryReadTimestamps(stamps, 0, t))
        {
            gpu.Add((t[1] - t[0]) / 1e6);
        }
    }

    var g = gpu.Count > 0 ? Median(gpu) : double.NaN;
    var w = Median(wall);
    Console.WriteLine(string.Create(CultureInfo.InvariantCulture, $"{name} | {g:F3} | {w:F3} | {bytes / (g * 1e6):F1} | {bytes}"));
}

static double Median(List<double> v)
{
    v.Sort();
    return v[v.Count / 2];
}
