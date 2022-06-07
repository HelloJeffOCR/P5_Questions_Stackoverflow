#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='tagging_api',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    platforms='any',
    install_requires=['Flask'],
)
