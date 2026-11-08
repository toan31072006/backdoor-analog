"""Create a NEW code-only PiW Cross/Nod ZIP; never install, upload or launch."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    files = []
    for path in (root / 'ATKBackd').rglob('*'):
        rel = path.relative_to(root)
        if any(part.startswith('.') or part in ('tests', '__pycache__', 'runs', 'envs') for part in rel.parts):
            continue
        if path.is_file() and (path.suffix in ('.py', '.json', '.yaml', '.yml')
                              or path.name.startswith(('LICENSE', 'NOTICE'))):
            files.append(path)
    files += [p for p in Path(__file__).resolve().parent.iterdir()
              if p.is_file() and p.suffix in ('.py', '.ps1', '.md')]
    files += [root / 'AGENTS.md', root / 'remote_linux/MMFI_CVPR_EXPERIMENT_PLAN_20261015.md']
    manifest = {p.relative_to(root).as_posix(): {
        'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size}
        for p in sorted(set(files))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in manifest:
            archive.write(root / name, name)
        archive.writestr('PIW_CROSS_NOD_CODE_MANIFEST.json', json.dumps(manifest, indent=2))
    print(f'CODE_ONLY_NO_EXPERIMENT_RUN: {args.output.resolve()} ({len(manifest)} files)')
    print('SHA256=' + hashlib.sha256(args.output.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
