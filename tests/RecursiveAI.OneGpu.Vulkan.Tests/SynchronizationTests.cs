namespace RecursiveAI.OneGpu.Vulkan.Tests;

/// <summary>
/// Staging memory that the device hands out again must be ordered after its earlier use by the queue itself, not only by the host having
/// waited. The fixture fails any test during which the validation layer, with synchronization checking on, reports a hazard.
/// </summary>
public sealed class SynchronizationTests
{
    /// <summary>The readback staging buffer of one size is reused after each read, so a copy into it follows earlier copies into it.</summary>
    [Fact]
    public async Task AsyncReadbacksThatReuseStagingAreOrderedAfterEachOther()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        var source = new uint[256];
        using var buffer = d.CreateBuffer<uint>(source);
        var back = new uint[source.Length];
        for (uint round = 0; round < 6; round++)
        {
            for (var i = 0; i < source.Length; i++)
            {
                source[i] = (round << 16) | (uint)i;
            }

            buffer.Write(source);
            await buffer.ReadAsync(back);
            Assert.Equal(source, back);
            Array.Clear(back);
        }
    }

    /// <summary>Reads that are in flight together each get their own staging buffer; finishing them in turn then reuses all of them.</summary>
    [Fact]
    public async Task OverlappingAsyncReadbacksAreOrderedAfterEarlierOnes()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var buffer = d.CreateBuffer<uint>(new uint[1024]);
        var results = new uint[4][];
        for (var round = 0; round < 3; round++)
        {
            var pending = new Task[results.Length];
            for (var k = 0; k < results.Length; k++)
            {
                buffer.Write(Enumerable.Repeat((uint)((round * 10) + k), 1024).ToArray());
                results[k] = new uint[1024];
                pending[k] = buffer.ReadAsync(results[k]).AsTask();
            }

            for (var k = 0; k < results.Length; k++)
            {
                await pending[k];
                Assert.All(results[k], v => Assert.Equal((uint)((round * 10) + k), v));
            }
        }
    }
}
