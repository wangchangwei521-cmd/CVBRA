"""Verify the release inventory without training, network or third-party imports."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    count = 0
    failures = []
    for line in (ROOT / 'FILE_MANIFEST_SHA256.txt').read_text(encoding='utf-8').splitlines():
        expected, name = line.split('  ', 1)
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            failures.append(name)
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            failures.append(name)
        count += 1
    if failures:
        raise SystemExit('FAIL: ' + ', '.join(failures))
    print(f'PASS: {count} release files match SHA-256 inventory.')

if __name__ == '__main__':
    main()
