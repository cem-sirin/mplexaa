from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

short_description = (
    "A package for computing the Multiplex Adamic-Adar score, a measure of node similarity in multiplex networks."
)

setup(
    name="mplexaa",
    version="0.1.2",
    packages=find_packages(),
    author="Cem Sirin",
    description=short_description,
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/cem-sirin/mplexaa",
    author_email="sirincem1@gmail.com",
    install_requires=requirements,
)
