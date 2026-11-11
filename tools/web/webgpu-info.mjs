// A page script for run-page.mjs (--script): reports what the browser's WebGPU offers, so a test run records its device.
//   node run-page.mjs --root <dir with any index.html> --script webgpu-info.mjs
export default async function ({ evalJs, sleep }) {
  // The page is still loading when the script starts; WebGPU exists only once the real page (a secure context) is there.
  for (let i = 0; i < 100 && !(await evalJs('!!navigator.gpu')); i++) await sleep(200);
  return await evalJs(`(async () => {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) return { adapter: null };
    const info = adapter.info || {};
    const limits = {};
    for (const k in adapter.limits) limits[k] = adapter.limits[k];
    return {
      userAgent: navigator.userAgent,
      info: { vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description, subgroupMinSize: info.subgroupMinSize, subgroupMaxSize: info.subgroupMaxSize },
      features: [...adapter.features].sort(),
      wgslLanguageFeatures: [...navigator.gpu.wgslLanguageFeatures].sort(),
      limits: Object.fromEntries(Object.entries(limits).filter(([k]) => /Compute|Storage|Buffer|BindGroups/.test(k))),
    };
  })()`);
}
