# pycanari
Open-source library for probabilistic anomaly detection in time series 

## Installation

#### Create Miniconda Environment

1. Install Miniconda by following these [instructions](https://docs.conda.io/en/latest/miniconda.html)
2. Create a conda environment:

    ```sh
    conda create --name pycanari python=3.10
    ```

3. Activate conda environment:

    ```sh
    conda activate pycanari
    ```

#### Install pycanari
1. Install pycanari

    ```sh
    pip install pycanari
    ```

2. [Search pycanari and download pycanari-0.0.1.tar.gz file from the lastest version](https://pypi.org)

3. Copy the downloaded pycanari-0.0.1.tar file to the your working folder

4. Extract the pycanari-0.0.1.tar file using:

    ```sh
    tar -xvf pycanari-0.0.1.tar
    ```
5. Set directory
    ```sh
    cd pycanari-0.0.1
    ```

6. Install requirements:

    ```sh
    pip install -r requirements.txt
    ```

7. Test **pycanari** package:

    ```sh
    python -m examples.toy_forecast
    ```

NOTE: Replace the name `pycanari-0.0.1` with the corresponding version, e.g. pycanari-0.0.2