from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'A package for processing and analysing data for Tom Jessel PHD'

# Setting up
setup(
    name="resources",
    version=VERSION,
    author="Tom Jessel",
    author_email="<jesselt@cardiff.ac.uk>",
    description=DESCRIPTION,
    packages=find_packages(),
)
