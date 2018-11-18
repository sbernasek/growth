from distutils.core import setup
from setuptools import find_packages

setup(
    name='growth',
    version='0.1',
    author='Sebastian Bernasek',
    author_email='sebastian@u.northwestern.com',
    packages=find_packages(exclude=('tests',)),
    scripts=[],
    url='https://github.com/sebastianbernasek/growth',
    license='MIT',
    description='Package for simulating mitotic recombination during tissue development.',
    long_description=open('README.md').read(),
    install_requires=[
        "matplotlib >= 2.0.0",
        "scipy >= 1.1.0",
        "networkx>=2.2",
        "ete3>=3.1.1",
        "PyQt5>=5.11.3"],
)
