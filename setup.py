from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="mplexaa",
    version="0.1",
    packages=find_packages(),
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/cem-sirin/mplexaa",
    author_email="sirincem1@gmail.com",
    install_requires=requirements,
)
