"""Package this serial two-attack launcher without data, models or credentials."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    'remote_tools/piw_metafi_two_extra_4090.py',
    'remote_tools/Start-PiWMetaFiTwoExtra4090.ps1',
    'remote_tools/PIW_METAFI_TWO_EXTRA_4090_VI.md',
    'remote_tools/piw_metafi_four_attacks_4090.py',
    'remote_tools/piw_metafi_a1_4090.py',
    'remote_tools/Start-PiWMetaFiA14090.ps1',
    'remote_tools/piw_metafi_clean_4090.py',
    'remote_tools/Start-PiWMetaFiClean4090.ps1',
    'remote_tools/piw_metafi_clean_vastenv_3090.py',
    'remote_tools/piw_metafi_remaining_3090/controller.py',
    'remote_tools/piw_metafi_remaining_3090/clean_profile.py',
    'remote_tools/piw_metafi_remaining_3090/source_snapshot.tar.gz',
    'remote_tools/piw_metafi_remaining_3090/source_plan.json.gz',
    'remote_tools/piw_metafi_remaining_3090/ftrojan_pattern_linux.f8.zlib',
    'tests/test_piw_metafi_two_extra_4090.py',
    'remote_tools/Build-PiWMetaFiTwoExtra4090.py',
)


def build(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Preserve existing package: {output}')
    contents = {name: (ROOT / name).read_bytes() for name in FILES}
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for name, data in contents.items():
            z.writestr(name, data)
        z.writestr('PACKAGE_MANIFEST.json', json.dumps(dict(
            profile='piw_metafi_two_extra_s42_4090_serial_v1',
            note='Code handoff only; no new full CUDA training results.',
            files=hashes), indent=2) + '\n')
    with zipfile.ZipFile(output) as z:
        if z.testzip() is not None or set(z.namelist()) != set(FILES) | {'PACKAGE_MANIFEST.json'}:
            raise ValueError('Package integrity/member verification failed')
        for name, expected in hashes.items():
            if hashlib.sha256(z.read(name)).hexdigest() != expected:
                raise ValueError('Packaged source checksum differs: ' + name)
    print(f'PACKAGE={output}')
    print(f'SHA256={hashlib.sha256(output.read_bytes()).hexdigest()}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    build(p.parse_args().out)
