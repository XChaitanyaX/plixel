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
)