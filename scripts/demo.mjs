import { delimiter, resolve } from 'node:path';
import { root, run } from './tooling.mjs';

const env = {
  ...process.env,
  TENSORLOOM_DEMO_DIR: resolve(root, 'artifacts/demo'),
  PYTHONPATH: [resolve(root, 'python'), process.env.PYTHONPATH].filter(Boolean).join(delimiter),
};
try {
  run('mix', ['run', '--no-compile', 'examples/demo.exs'], { env });
  run(resolve(root, '.venv/bin/python'), ['examples/demo.py'], { env });
} catch (error) {
  console.error(`${error.message}\nRun npm run setup before running the demo.`);
  process.exitCode = 1;
}
