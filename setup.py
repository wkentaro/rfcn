#!/usr/bin/env python

from setuptools import find_packages
from setuptools import setup


version = '0.0.1'


setup(
    name='rfcn',
    version=version,
    packages=find_packages(),
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
