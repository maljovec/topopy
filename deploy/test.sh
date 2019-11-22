#!/bin/bash
set -e
set -x

# Append the version number with this git commit hash
GIT_HASH=$(git rev-parse --short HEAD)
gawk -i -v hash=$GIT_HASH '/^__version__ = \"/{ sub(/"$/,"-"hash"&") }1' topopy/__init__.py
TEMP_VERSION=$(grep  '__version__ = ' topopy/__init__.py | cut -d = -f 2)
echo $TEMP_VERSION

# Build the project
make
python setup.py sdist

# Test the upload
TWINE_USERNAME=__twine__
TWINE_PASSWORD=$PYPI_TOKEN
twine upload --repository-url https://test.pypi.org/legacy/ dist/topopy-$TEMP_VERSION.tar.gz
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple topopy==TEMP_VERSION
python -c "import topopy"
