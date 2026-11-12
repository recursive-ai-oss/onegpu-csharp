using System.Runtime.InteropServices.JavaScript;
using System.Runtime.Versioning;

namespace RecursiveAI.OneGpu.WebGpu;

/// <summary>The functions of <c>onegpu.js</c> (every one but <c>init</c>, <c>setCanvasSize</c> and <c>takeRead</c> names the device's context first, so several devices can share a page), registered as the module "onegpu" by the page or, when it did not, by <c>WebGpu.CreateDeviceAsync</c>.</summary>
[SupportedOSPlatform("browser")]
internal static partial class Js
{
    private const string Module = "onegpu";

    [JSImport("init", Module)]
    [return: JSMarshalAs<JSType.Promise<JSType.String>>]
    public static partial Task<string> Init(string requestJson);

    [JSImport("exec", Module)]
    public static partial void Exec(int context, [JSMarshalAs<JSType.MemoryView>] Span<byte> stream);

    [JSImport("takeMessages", Module)]
    public static partial string TakeMessages(int context);

    [JSImport("configureCanvas", Module)]
    public static partial string ConfigureCanvas(int context, string canvasId, int width, int height);

    [JSImport("setCanvasSize", Module)]
    public static partial void SetCanvasSize(string canvasId, int width, int height);

    // A read finishes with a ticket; the bytes are then copied out of the script with TakeRead.
    [JSImport("whenIdle", Module)]
    [return: JSMarshalAs<JSType.Promise<JSType.Void>>]
    public static partial Task WhenIdle(int context);

    [JSImport("readTexture", Module)]
    [return: JSMarshalAs<JSType.Promise<JSType.Number>>]
    public static partial Task<int> ReadTexture(int context, int id, int mip, int layer, int width, int height, int depth, int bytesPerTexel);

    [JSImport("readBuffer", Module)]
    [return: JSMarshalAs<JSType.Promise<JSType.Number>>]
    public static partial Task<int> ReadBuffer(int context, int id, double offset, double size);

    [JSImport("takeRead", Module)]
    public static partial void TakeRead(int ticket, [JSMarshalAs<JSType.MemoryView>] Span<byte> destination);
}
