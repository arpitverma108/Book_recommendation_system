from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

## edit below variables as per your requirements -
REPO_NAME = "book recommendation system"
AUTHOR_USER_NAME = "Arpit Verma"
SRC_REPO = "books_recommendation_system"
LIST_OF_REQUIREMENTS = []


setup(
    name=SRC_REPO,
    version="0.0.1",
    author="Arpit Verma",
    description="A small local packages for ML based books recommendations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/arpitverma108/Book_recommendation_system.git",
    author_email="arpitv0710@gmail.com",
    packages=find_packages(),
    license="MIT",
    python_requires=">=3.9",
    install_requires=LIST_OF_REQUIREMENTS
)