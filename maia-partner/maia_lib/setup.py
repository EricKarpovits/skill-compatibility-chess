import re

import setuptools

with open("maia_lib/utils.py") as f:
    versionString = re.search(r'__version__ = "(.+)"', f.read()).group(1)

with open("README.md") as f:
    readme = f.read()

with open("requirements.txt") as f:
    requirements = [l.strip() for l in f]

if __name__ == "__main__":
    setuptools.setup(
        name="maia_lib",
        version=versionString,
        author="x",
        author_email="x",
        description="x",
        url="x",
        long_description=readme,
        long_description_content_type="text/markdown",
        zip_safe=False,
        python_requires=">=3.7",
        install_requires=requirements,
        packages=["maia_lib"] + [f"maia_lib.{p}" for p in setuptools.find_packages('maia_lib')],
        include_package_data=True,
        package_data={
            "model_utils/maia": ["*.pb.gz"],
            "model_utils/stockfish": ["stockfish_14*"],
        },
    )
