#!/bin/bash

# Append the version number with this git commit hash
GIT_HASH=$(git rev-parse --short HEAD)
awk -v hash=$GIT_HASH '/^__version__ = \"/{ sub(/"$/,"-"hash"&") }1' topopy/__init__.py > topopy/__init__.py
TEMP_VERSION=$(grep  '__version__ = ' topopy/__init__.py | cut -d = -f 2)

# Build the project
make
python setup.py sdist

# Test the upload
twine upload --repository-url https://test.pypi.org/legacy/ -u __twine__ -p $PYPI_TOKEN dist/topopy-$TEMP_VERSION.tar.gz
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple topopy==TEMP_VERSION
python -c "import topopy"
