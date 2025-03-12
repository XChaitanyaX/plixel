from setuptools import setup, find_packages

setup(
    name='plixel',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'openpyxl',
        'setuptools',
        'matplotlib',
        'seaborn',
    ],
    description="A package to analyse excel and csv files",
    author="Bonu Krishna Chaitanya",
    author_email="bkc14042005@gmail.com",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
)