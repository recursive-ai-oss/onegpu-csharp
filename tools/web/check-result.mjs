// Prints the summary of a browser test run (the JSON the web runner reports) and exits non-zero when anything failed.
//   node check-result.mjs result.json
import fs from 'node:fs';

const r = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
if (r.fatal) {
  console.error('fatal: ' + r.fatal);
  process.exit(1);
}
for (const t of r.tests || []) {
  if (t.status === 'failed') console.error('FAILED ' + t.name + '\n' + t.error);
  else if (t.status === 'skipped') console.log('skipped ' + t.name + ': ' + t.reason);
}
for (const n of r.notes || []) console.log(n);
if (r.capabilities) console.log(r.adapter + ' compute capabilities:\n' + r.capabilities);
console.log(`${r.adapter}: ${r.passed} passed, ${r.failed} failed, ${r.skipped} skipped`);
process.exit(r.failed > 0 || r.passed === 0 ? 1 : 0);
