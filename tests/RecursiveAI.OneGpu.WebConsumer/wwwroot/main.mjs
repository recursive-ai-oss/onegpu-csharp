// Starts the runtime and calls the app. It does not import onegpu.js or register it: the package's script is loaded by the library.
import { dotnet } from './_framework/dotnet.js';

try {
  const runtime = await dotnet.create();
  const exports = await runtime.getAssemblyExports(runtime.getConfig().mainAssemblyName);
  const json = await exports.Run.Go();
  window.__result = json;
  document.getElementById('log').textContent = json;
} catch (e) {
  window.__result = JSON.stringify({ fatal: String(e && e.stack || e) });
  document.getElementById('log').textContent = 'fatal: ' + e;
}
