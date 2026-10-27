// Released CLI composition: ref:DEC-build-artifacts. Keep review status separate.
import { mkdirSync, writeFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { resolve } from 'node:path';
const root = fileURLToPath(new URL('..', import.meta.url));
const cli = resolve(root, 'node_modules/lawspec/bin/lawspec.mjs');
const canonCli = resolve(root, 'node_modules/ricestax/bin/ricestax.mjs');
const summary = { components: {}, canon: { status: 'not-run' } };
mkdirSync(resolve(root, 'artifacts'), { recursive: true });
function run(label, args, allowFailure = false, executable = cli) {
  console.log(label);
  const result = spawnSync(process.execPath, [executable, ...args], {
    cwd: root, encoding: 'utf8', maxBuffer: 64 * 1024 * 1024,
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  writeFileSync(resolve(root, `artifacts/${label}.${args.includes('--json') ? 'json' : 'log'}`), result.stdout || '');
  writeFileSync(resolve(root, `artifacts/${label}.stderr.log`), result.stderr || '');
  if (result.error || result.signal || (result.status !== 0 && !allowFailure)) {
    process.stderr.write(result.stdout || '');
    process.stderr.write(result.stderr || String(result.error || result.signal || ''));
    process.exit(1);
  }
  return args.includes('--json') ? JSON.parse(result.stdout) : result;
}
for (const [component, config, target] of [
  ['elixir', 'lawspec.edn', 'elixir'],
  ['pytorch', 'python/lawspec.edn', 'python'],
]) {
  run(`${component}-definitions`, ['check', '--config', config, '--json']);
  run(`${component}-generation`, ['generate', '--config', config, '--target', target]);
  const tests = run(`${component}-native`, ['test', '--config', config, '--target', target, '--fresh', '--json']);
  if (!tests.every(test => test.ok)) process.exit(1);
  summary.components[component] = { definitions: 'passed', native: 'passed', results: tests };
  console.log(`${component}: LawSpec native execution passed`);
}
// LawSpec 0.31.0 has no project status/validate verbs. Use released Canon
// interfaces directly rather than depending on an unpublished development CLI.
run('canon-review', ['canon', 'vet'], false, canonCli);
const canon = run('canon-validation', ['canon', 'check'], true, canonCli);
if (canon.status !== 0 && canon.status !== 1) {
  process.stderr.write(canon.stdout || '');
  process.stderr.write(canon.stderr || 'Canon could not run.');
  process.exit(1);
}
summary.canon = {
  status: canon.status === 0 ? 'passed' : 'nonpassing',
  exitCode: canon.status,
  reviewReport: 'artifacts/canon-review.log',
  validationReport: 'artifacts/canon-validation.log',
};
writeFileSync(resolve(root, 'artifacts/verification-summary.json'), JSON.stringify(summary, null, 2) + '\n');
if (canon.status !== 0) {
  console.log('Executable checks passed. Canon validation remains nonpassing; see artifacts/canon-review.log and artifacts/canon-validation.log for pending review and any findings.');
  process.exitCode = 2;
} else console.log('Both LawSpec components and Canon validation passed.');
