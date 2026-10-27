import { components, lawspec, run } from './tooling.mjs';
for (const [, config, target] of components) {
  run(process.execPath, [lawspec, 'generate', '--config', config, '--target', target]);
}
