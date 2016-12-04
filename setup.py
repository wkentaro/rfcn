#!/usr/bin/env python

import distutils

import Cython.Build
import numpy
from setuptools import find_packages
from setuptools import setup


version = '0.0.1'

ext_modules = [
    distutils.extension.Extension(
        'rfcn.external.faster_rcnn.faster_rcnn.bbox',
        ['rfcn/external/faster_rcnn/faster_rcnn/bbox.pyx'],
        include_dirs=[numpy.get_include()]
    ),
    distutils.extension.Extension(
        'rfcn.external.faster_rcnn.faster_rcnn.cpu_nms',
        ['rfcn/external/faster_rcnn/faster_rcnn/cpu_nms.pyx'],
        include_dirs=[numpy.get_include()]
    ),
]

setup(
    name='rfcn',
    version=version,
    packages=find_packages(),
    ext_modules=Cython.Build.cythonize(ext_modules),
    install_requires=open('requirements.txt').readlines(),
    description='Recurrent Fully Convolutional Networks',
    long_description=open('README.rst').read(),
    author='Kentaro Wada',
    author_email='www.kentaro.wada@gmail.com',
    url='http://github.com/wkentaro/rfcn',
    license='MIT',
    keywords='machine-learning',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX',
        'Topic :: Internet :: WWW/HTTP',
    ],
)
