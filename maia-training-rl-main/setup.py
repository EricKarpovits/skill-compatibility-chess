import setuptools
import re

with open('maia_rl/__init__.py') as f:
    versionString = re.search(r"__version__ = '(.+)'", f.read()).group(1)

if __name__ == '__main__':
    setuptools.setup(
        name='maia_rl',
        version = versionString,
        author="x",
        author_email="x",
        zip_safe = True,
        packages = [
                'maia_rl',
                'maia_rl.leela_board',
                'maia_rl.proto',
                'maia_rl.file_handling',
                'maia_rl.pgn_handling',
                'maia_rl.tf',
            ],
    )
