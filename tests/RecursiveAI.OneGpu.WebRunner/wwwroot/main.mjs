import { dotnet } from './_framework/dotnet.js';
import * as gpu from './onegpu.js';

const log = document.getElementById('log');
try {
  const filter = new URLSearchParams(location.search).get('filter') || '';
  const runtime = await dotnet.withDiagnosticTracing(false).create();
  runtime.setModuleImports('onegpu', gpu);
  const config = runtime.getConfig();
  const exports = await runtime.getAssemblyExports(config.mainAssemblyName);
  const json = await exports.RecursiveAI.OneGpu.Vulkan.Tests.Runner.Run(filter);
  window.__result = json;
  const r = JSON.parse(json);
  log.textContent = `${r.adapter}: ${r.passed} passed, ${r.failed} failed, ${r.skipped} skipped`;
} catch (e) {
  window.__result = JSON.stringify({ fatal: String(e && e.stack || e) });
  log.textContent = 'fatal: ' + e;
}
