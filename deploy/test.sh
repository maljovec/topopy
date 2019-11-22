#!/bin/bash
set -e
set -x

# Append the version number with this git commit hash, but hashes contain
# letters which are not allowed in pypi versions. We can hack this to replace
# all letters with numbers, this should still be unique enough to not collide
# before the version number increases.
GIT_HASH=$(git rev-parse --short HEAD | tr 'abcdefghijklmnopqrstuvwxyz' '12345678901234567890123456')
awk -v hash=$GIT_HASH '/^__version__ = \"/{ sub(/"$/,".dev"hash"&") }1' topopy/__init__.py > tmp && mv tmp topopy/__init__.py
TEMP_VERSION=$(grep  '__version__ = ' topopy/__init__.py | cut -d = -f 2 | sed "s/\"//g" | sed 's/^[ \t]*//;s/[ \t]*$//')
TEMP_VERSION=$(expr $TEMP_VERSION)
echo $TEMP_VERSION

# Build the project
make
python setup.py sdist

# Test the upload# Test the upload, temporarily disable exit on error, since there is a race 
# condition for which build will get this out first, also, re-triggered builds
# would never succeed in this step.
set +e
twine upload --repository-url https://test.pypi.org/legacy/ -u __token__ -p ${PYPI_TOKEN} --non-interactive dist/topopy-${TEMP_VERSION}.tar.gz
set -e

#Give it some time to register internally before trying to install it
sleep 60

# Now install and run it
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple topopy==${TEMP_VERSION}
python -c "import topopy"
