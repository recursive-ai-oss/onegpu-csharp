namespace RecursiveAI.OneGpu.Kernels.Tests;

/// <summary>More than one device in one process or page: a fallback session after a refused one, an encoder and a decoder side by side. Each device has its own buffers and push-constant storage.</summary>
public sealed class MultiDeviceTests
{
    [Fact]
    public async Task DevicesDoNotShareState()
    {
        using var gpu = new GpuFixture();
        var first = gpu.Device;
        var second = await GpuFixture.NewDeviceAsync();
        var third = await GpuFixture.NewDeviceAsync();
        try
        {
            var devices = new[] { first, second, third };
            var kernels = devices.Select(d => new ElementwiseKernels(d)).ToArray();
            var gemms = devices.Select(d => new Int8Gemm(d)).ToArray();
            try
            {
                // Interleave recording and submission so each device's pass state, push constants and handles are used between the others' work.
                var x = new float[1000];
                for (var i = 0; i < x.Length; i++)
                {
                    x[i] = i % 17;
                }

                var inputs = devices.Select(d => d.CreateBuffer<float>(x)).ToArray();
                var outputs = devices.Select(d => d.CreateBuffer<float>(x.Length)).ToArray();
                var sums = devices.Select(d => d.CreateBuffer<int>(6)).ToArray();
                var a = devices.Select(d => d.CreateBuffer<byte>([1, 2, 3, 4, 5, 6])).ToArray();
                var b = devices.Select(d => d.CreateBuffer<sbyte>([1, -1, 2, -2, 3, -3])).ToArray();
                try
                {
                    for (var round = 0; round < 3; round++)
                    {
                        for (var i = 0; i < devices.Length; i++)
                        {
                            var enc = devices[i].BeginCommands();
                            kernels[i].EncodeBinary(enc, GpuBinaryOp.Add, inputs[i], 0, [x.Length], inputs[i], 0, [x.Length], outputs[i], 0);
                            gemms[i].EncodeInt32(enc, new Int8GemmProblem(2, 3, 2), a[i], b[i], sums[i]);
                            devices[i].Submit(enc);
                        }
                    }

                    for (var i = 0; i < devices.Length; i++)
                    {
                        var got = await Rig.DownloadAsync(outputs[i]);
                        for (var j = 0; j < x.Length; j++)
                        {
                            Assert.Equal(2 * x[j], got[j]);
                        }

                        // [[1 2], [3 4]] x [[1 -1 2], [-2 3 -3]] = [[-3 5 -4], [-5 9 -6]]
                        var s = await Rig.DownloadAsync(sums[i], 6);
                        Assert.Equal(new[] { -3, 5, -4, -5, 9, -6 }, s);
                    }
                }
                finally
                {
                    foreach (var buffer in inputs.Concat(outputs))
                    {
                        buffer.Dispose();
                    }

                    foreach (var buffer in sums)
                    {
                        buffer.Dispose();
                    }

                    foreach (var buffer in a)
                    {
                        buffer.Dispose();
                    }

                    foreach (var buffer in b)
                    {
                        buffer.Dispose();
                    }
                }
            }
            finally
            {
                foreach (var k in kernels)
                {
                    k.Dispose();
                }

                foreach (var g in gemms)
                {
                    g.Dispose();
                }
            }

            // A device that goes away leaves the others working.
            second.WaitIdle();
            Assert.Empty(second.TakeValidationMessages());
        }
        finally
        {
            foreach (var d in new[] { second, third })
            {
                d.WaitIdle();
                Assert.Empty(d.TakeValidationMessages());
                d.Dispose();
            }
        }
    }
}
