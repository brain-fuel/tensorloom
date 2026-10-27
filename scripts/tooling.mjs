// Pinned local tooling and source-only builds: ref:DEC-build-artifacts.
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { resolve } from 'node:path';

export const root = fileURLToPath(new URL('..', import.meta.url));
export const lawspec = resolve(root, 'node_modules/lawspec/bin/lawspec.mjs');
export const components = [
  ['elixir', 'lawspec.edn', 'elixir'],
  ['pytorch', 'python/lawspec.edn', 'python'],
];

export function run(command, args, options = {}) {
  console.log(`> ${command} ${args.join(' ')}`);
  const result = spawnSync(command, args, {
    cwd: root, env: process.env, stdio: 'inherit', ...options,
  });
  if (result.error || result.signal || result.status !== 0) {
    throw new Error(result.error?.message || `${command} failed (${result.signal || result.status})`);
  }
}
