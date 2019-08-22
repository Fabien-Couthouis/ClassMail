from setuptools import setup, find_packages
import site
import requests
import zipfile
import io
import tempfile
with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="classmail",
    version="0.1",
    description="A simple framework for automatise mail classification task",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Fabien Couthouis",
    author_email="fcouthouis@ensc.fr",
    url="https://github.com/Fabien-Couthouis/ClassMail",
    packages=find_packages(exclude="tests"),
    license="MIT",
    install_requires=required,
    data_files=[('config', ['classmail/nlp/conf.json'])],
    include_package_data=True,
    python_requires=">=3.6",
)
