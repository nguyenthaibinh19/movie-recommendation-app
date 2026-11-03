"""Download MovieLens 1M from GroupLens for local research use.

Read the dataset terms before use: https://files.grouplens.org/datasets/movielens/ml-1m-README.txt
Only the four named dataset files are read from the official archive.
Existing local files are preserved.
"""
import io
import os
from pathlib import Path
import shutil
import subprocess
from urllib.request import urlopen
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'Data' / 'Dataset'
FILES = ('movies.dat', 'users.dat', 'ratings.dat', 'README')
URL = 'https://files.grouplens.org/datasets/movielens/ml-1m.zip'


def main():
    missing = [name for name in FILES if not (DEST / name).is_file()]
    if not missing:
        print('MovieLens 1M is already present.')
        return
    print(f'Downloading the official MovieLens 1M archive: {URL}')
    # Windows' bundled curl uses the OS certificate store, including intermediate
    # certificate retrieval that some standalone Python installations lack.
    curl = shutil.which('curl.exe') if os.name == 'nt' else None
    if curl:
        payload = subprocess.run(
            [curl, '--fail', '--silent', '--show-error', '--location',
             '--max-time', '60', '--max-filesize', '10485760', URL],
            capture_output=True, check=True,
        ).stdout
    else:
        with urlopen(URL, timeout=60) as response:
            payload = response.read(10 * 1024 * 1024 + 1)
    if len(payload) > 10 * 1024 * 1024:
        raise ValueError('Dataset archive exceeds the expected size limit')
    with ZipFile(io.BytesIO(payload)) as archive:
        # Validate all members we need before writing any files.
        contents = {name: archive.read(f'ml-1m/{name}') for name in missing}
    DEST.mkdir(parents=True, exist_ok=True)
    for name, data in contents.items():
        target = DEST / name
        # Never overwrite a user's existing dataset.
        try:
            with target.open('xb') as output:
                output.write(data)
        except FileExistsError:
            pass
    print(f'MovieLens 1M is ready at {DEST}')


if __name__ == '__main__':
    main()
