# extrack-scripts
Contains scripts to use the [ExTrack](https://github.com/vanTeeffelenLab/ExTrack) python library.

The scripts provide command line tools to run ExTrack on traced localisation files
in CSV format, save the arguments and fitted models, and run predictions on the
data using the ExTrack model.

# Install

Setup the environment as described on the ExTrack documentation. This recommends
using a conda enviroment to install dependencies (`cupy` is optional).

    conda create -n extrack -c conda-forge python pip numpy lmfit xmltodict matplotlib pandas jupyter # cupy
    conda activate extrack

Clone the ExTrack repository and install in the enviroment.
The ExTrack documentation uses the deprecated `python setup.py install`.
This can be replaced with:

    [path/to/ExTrack] $ pip install .

The extrack-scripts in this repository can then be added to the PATH.

- run-extrack.py: runs ExTrack on input CSV files, saves the fitted model,
  and outputs predicted states and positions

- show-model.py: shows details of fitted models
