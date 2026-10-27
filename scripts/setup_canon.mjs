// Recreate only the ignored grammar cache; preserve all project Canon metadata.
import { access, cp, mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { initialize } from '../../lawspec/npm/node_modules/ricestax/setup.mjs';
const root = fileURLToPath(new URL('..', import.meta.url));
const destination = join(root, '.ricestax');
try {
  await access(join(destination, 'grammars/folio/FolioParser.g4'));
  console.log('Canon grammar cache is present.');
} catch {
  const temporary = await mkdtemp(join(tmpdir(), 'tensorloom-canon-'));
  try {
    await initialize(temporary, ['folio']);
    await cp(join(temporary, '.ricestax'), destination, { recursive: true });
    console.log('Canon grammar cache restored; review metadata preserved.');
  } finally {
    await rm(temporary, { recursive: true, force: true });
  }
}
