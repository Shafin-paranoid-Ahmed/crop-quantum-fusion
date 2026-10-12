"""Fetch frozen inputs from a local companion or the private baseline release."""
import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def install(source, item):
    destination = ROOT / item['path']
    if destination.is_file():
        if digest(destination) == item['sha256']:
            print('Already verified:', item['path'])
            return
        raise RuntimeError(f'Refusing to overwrite a different artifact: {destination}')
    if digest(source) != item['sha256']:
        raise RuntimeError(f'Checksum mismatch: {source}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    assert digest(destination) == item['sha256']
    print('Installed:', item['path'])

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--from-directory', type=Path, help='Baseline checkout containing the frozen files')
    group.add_argument('--from-archive', type=Path, help='Local embeddings-v1.zip bundle')
    group.add_argument('--download', action='store_true', help='Use authenticated gh to download the private release')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'ARTIFACTS.json').read_text(encoding='utf-8'))
    if args.from_directory:
        for item in manifest['files']:
            install(args.from_directory / item['path'], item)
        return
    if args.download and not manifest.get('published', False):
        raise RuntimeError('This snapshot is local-only. Use --from-directory or --from-archive; no GitHub release has been published.')
    with tempfile.TemporaryDirectory(prefix='crop-artifacts-') as temporary:
        directory = Path(temporary)
        if args.download:
            subprocess.run(['gh', 'release', 'download', manifest['release'], '--repo', manifest['repository'], '--pattern', manifest['asset'], '--dir', str(directory)], check=True)
        archive_path = args.from_archive if args.from_archive else directory / manifest['asset']
        with zipfile.ZipFile(archive_path) as archive:
            for item in manifest['files']:
                # Extract only allowlisted paths into a flat temporary file.
                candidate = directory / 'verified-input.npz'
                with archive.open(item['path']) as source, candidate.open('wb') as output:
                    shutil.copyfileobj(source, output)
                install(candidate, item)

if __name__ == '__main__':
    main()
