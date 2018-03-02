########################################################################
# Software License Agreement (BSD License)                             #
#                                                                      #
# Copyright 2018 University of Utah                                    #
# Scientific Computing and Imaging Institute                           #
# 72 S Central Campus Drive, Room 3750                                 #
# Salt Lake City, UT 84112                                             #
#                                                                      #
# THE BSD LICENSE                                                      #
#                                                                      #
# Redistribution and use in source and binary forms, with or without   #
# modification, are permitted provided that the following conditions   #
# are met:                                                             #
#                                                                      #
# 1. Redistributions of source code must retain the above copyright    #
#    notice, this list of conditions and the following disclaimer.     #
# 2. Redistributions in binary form must reproduce the above copyright #
#    notice, this list of conditions and the following disclaimer in   #
#    the documentation and/or other materials provided with the        #
#    distribution.                                                     #
# 3. Neither the name of the copyright holder nor the names of its     #
#    contributors may be used to endorse or promote products derived   #
#    from this software without specific prior written permission.     #
#                                                                      #
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR #
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED       #
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE   #
# ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY       #
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL   #
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE    #
# GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS        #
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER #
# IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR      #
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN  #
# IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                        #
########################################################################
"""
      Setup script for topopy
"""
from setuptools import setup, Extension
import re


def get_property(prop, project):
    """
        Helper function for retrieving properties from a project's
        __init__.py file
        @In, prop, string representing the property to be retrieved
        @In, project, string representing the project from which we will
        retrieve the property
        @Out, string, the value of the found property
    """
    result = re.search(r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(prop),
                       open(project + '/__init__.py').read())
    return result.group(1)


FILES = ['UnionFind.cpp', 'MergeTree.cpp', 'AMSC.cpp', 'utils.cpp',
         'topology_wrap.cpp']
VERSION = get_property('__version__', 'topopy')

# Consult here: https://packaging.python.org/tutorials/distributing-packages/
setup(name='topopy',
      packages=['topopy'],
      version=VERSION,
      description='A library for computing topological data structures',
      long_description='Given a set of arbitrarily arranged points in any ' +
                       'dimension, this library is able to construct ' +
                       'approximate topological structures using a ' +
                       'neighborhood graph.',
      author='Dan Maljovec',
      author_email='maljovec002@gmail.com',
      license='BSD',
      test_suite='topopy.tests',
      url='https://github.com/maljovec/topopy',
      download_url='https://github.com/maljovec/topopy/archive/' + VERSION +
                   '.tar.gz',
      keywords=['topological data analysis', 'computational topology',
                'Morse theory', 'merge tree', 'contour tree', 'extremum graph',
                'Morse-Smale complex', 'Morse complex'],
      # Consult here: https://pypi.python.org/pypi?%3Aaction=list_classifiers
      classifiers=['Development Status :: 3 - Alpha',
                   'Intended Audience :: Science/Research',
                   'License :: OSI Approved :: BSD License',
                   'Programming Language :: C++',
                   'Programming Language :: Python :: 2',
                   'Programming Language :: Python :: 3',
                   'Topic :: Scientific/Engineering :: Mathematics'],
      install_requires=['numpy', 'scipy', 'scikit-learn', 'networkx',
                        'nglpy>=1.0.1'],
      python_requires='>=2.7, <4',
      # package_dir={'':'src/'},
      ext_modules=[Extension('_topology', FILES)])
