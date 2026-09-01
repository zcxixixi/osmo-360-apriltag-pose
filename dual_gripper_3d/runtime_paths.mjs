import fs from 'node:fs';


const DEFAULT_CHROME_CANDIDATES = [
  '/usr/bin/google-chrome',
  '/usr/bin/google-chrome-stable',
  '/usr/bin/chromium',
  '/usr/bin/chromium-browser',
  '/snap/bin/chromium',
];


function executable(path) {
  try {
    fs.accessSync(path, fs.constants.X_OK);
    return true;
  } catch {
    return false;
  }
}


export function resolveChromeExecutable({argument, environment = process.env} = {}) {
  const explicit = argument || environment.OSMO_CHROME_BINARY;
  if (explicit) {
    if (!executable(explicit)) {
      throw new Error(`configured Chrome binary is missing or not executable: ${explicit}`);
    }
    return explicit;
  }
  const detected = DEFAULT_CHROME_CANDIDATES.find(executable);
  if (detected) return detected;
  throw new Error(
    'no Chrome/Chromium runtime found; set --chrome or OSMO_CHROME_BINARY to an executable path',
  );
}
