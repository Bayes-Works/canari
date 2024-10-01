from setuptools import find_packages, setup
import os

# Function to read the requirements from requirements.txt
def read_requirements(filename='requirements.txt'):
    """Read requirements from a file and return as a list."""
    with open(os.path.join(os.path.dirname(__file__), filename), 'r') as f:
        lines = f.readlines()
    # Filter out comments and empty lines
    requirements = [line.strip() for line in lines if line.strip() and not line.startswith('#')]
    return requirements

setup(
    name='canari',
    version='0.1',
    description='Probabilistic anomaly detection in time series',
    author="Van-Dai Vuong, Luong-Ha Nguyen, James-A. Goulet",
    author_email="vuongdai@gmail.com, luongha.nguyen@gmail.com, james.goulet@polymtl.ca",
    packages=find_packages('src'),
    package_dir={'': 'src'},
    include_package_data=True,
    zip_safe=False,
    install_requires=read_requirements(),  # Read from requirements.txt
)
