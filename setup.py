# setup.py
from setuptools import setup, find_packages

setup(
    name="canari",
    version="0.1",
    # packages=find_packages(where="src"),
    # package_dir={"": "src"},  # Specify the src directory
    install_requires=[
        "pyTAGI",  
        "matplotlib==3.8.2",
        "ninja==1.11.1.1",
        "numpy==1.26.0",
        "pandas==2.1.4",
        "tqdm==4.66.1",
        "fire==0.5.0",
    ],
    description="A Python package that uses pyTAGI",
    author="Van-Dai Vuong, James-A. Goulet",
    author_email="vuongdai@gmail.com"
)
