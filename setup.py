from distutils.core import setup
from setuptools import find_packages
from os import path

# read the contents of your README file
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='growth',
    version='v0.1',
    author='Sebastian Bernasek',
    author_email='sebastian@u.northwestern.com',
    packages=find_packages(exclude=('tests', 'scripts', 'sweep')),
    scripts=[],
    url='https://github.com/sebastianbernasek/growth',
    license='MIT',
    description='Package for simulating tissue growth.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    python_requires='>=3',
    install_requires=[
        "matplotlib >= 2.0.0",
        "scipy >= 1.1.0",
        "networkx>=2.2",
        "pandas>=0.23.4"],
    tests_require=['nose'],
    test_suite='nose.collector'
)
