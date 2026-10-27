// Execute both independently built LawSpec components; retain review status.
import { mkdirSync, writeFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { resolve } from 'node:path';
const root = fileURLToPath(new URL('..', import.meta.url));
const cli = resolve(root, '../lawspec/npm/bin/lawspec.mjs');
mkdirSync(resolve(root, 'artifacts'), { recursive: true });
function run(label, args, allowFailure = false) {
  console.log(label);
  const result = spawnSync(process.execPath, [cli, ...args], {
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
  console.log(`${component}: LawSpec native execution passed`);
}
run('project-status', ['project', 'status', '--project', '.', '--json']);
const project = run('project-validation', ['project', 'validate', '--project', '.', '--json'], true);
if (project.definitions.status !== 'passed' || project.verification.status !== 'completed' || project.native.status !== 'passed') process.exit(1);
if (project.canon.status !== 'passed') {
  console.log('Executable checks passed. Canon validation remains nonpassing; see artifacts/project-validation.json for pending review and any findings.');
  process.exitCode = 2;
} else console.log('Both LawSpec components and project validation passed.');
