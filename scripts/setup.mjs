import { resolve } from 'node:path';
import { root, run } from './tooling.mjs';

try {
  run('uv', ['sync', '--locked']);
  run('mix', ['deps.get']);
  run('mix', ['deps.compile'], { env: { ...process.env, MIX_ENV: 'test' } });
  run(process.execPath, [resolve(root, 'scripts/generate.mjs')]);
  run('mix', ['compile']);
  run(process.execPath, [resolve(root, 'scripts/setup_canon.mjs')]);
  console.log('Tensorloom is ready. Run npm run demo.');
} catch (error) {
  console.error(error.message);
  process.exitCode = 1;
}
