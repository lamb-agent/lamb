import { Bash, InMemoryFs } from 'just-bash';
import { readFileSync } from 'node:fs';

interface InputPayload {
  command: string;
  files: Record<string, string>;
}

interface OutputPayload {
  output: string;
  files: Record<string, string>;
}

/**
 * Functional-style executor using just-bash virtual environment
 */
const executeVirtualBash = async (): Promise<void> => {
  const home = "/home/user/";
  try {
    // read command and files from stdin
    const rawInput = readFileSync(0, 'utf-8');
    const { command, files }: InputPayload = JSON.parse(rawInput);

    // mount the files in the home dir
    const mountedFiles = Object.fromEntries(Object.entries(files).map(([path, content]) => [home + path, content]))

    const fs = new InMemoryFs(mountedFiles);
    const bash = new Bash({ fs, cwd: home });

    const result = await bash.exec(command);

    // extract modified files from home dir
    const allPaths = fs.getAllPaths().filter((path) => path.startsWith(home))
    const newFiles: Record<string, string> = Object.fromEntries(
      await Promise.all(
        allPaths.map(async (path) => {
          const stats = await fs.stat(path);
          if (!stats.isFile) return null; // Filter out directories
          return [path.slice(home.length), await fs.readFile(path, 'utf-8')];
        })
      ).then(results => results.filter((entry): entry is [string, string] => entry !== null))
    );

    // send output and files to stdout
    const response: OutputPayload = {
      output: result.stdout + result.stderr,
      files: newFiles,
    };

    process.stdout.write(JSON.stringify(response));
  } catch (err) {
    process.stderr.write(JSON.stringify({ error: String(err) }));
    process.exit(1);
  }
};

executeVirtualBash();
