import { readFile } from 'node:fs/promises';
import { parseOpenAiCompletionOutput, parseZincServerOutput } from './performance_suite.mjs';

const [engine, bodyFile, logFile] = process.argv.slice(2);
const body = await readFile(bodyFile, 'utf8');
const log = engine === 'zinc' ? await readFile(logFile, 'utf8') : '';
const parsed =
  engine === 'zinc'
    ? parseZincServerOutput(`${body}\n__ZINC_TIMING__\n${log}`)
    : parseOpenAiCompletionOutput(body);
process.stdout.write(JSON.stringify(parsed));
