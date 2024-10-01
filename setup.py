from setuptools import find_packages, setup

setup(
    name='canari',
    version='0.1',
    description='Probabilistic anomaly detection in time series',
    author="Van-Dai Vuong, James-A. Goulet, Luong-Ha Nguyen",
    author_email="vuongdai@gmail.com, james.goulet@polymtl.ca",
    packages=find_packages('src'),
    package_dir={'': 'src'},
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        "pyTAGI",
        "matplotlib==3.8.2",
        "ninja==1.11.1.1",
        "numpy==1.26.0",
        "pandas==2.1.4",
        "tqdm==4.66.1",
        "fire==0.5.0",
        "scikit-learn==1.3.0"
    ],
)