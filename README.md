# GP-age

GP-age is an epigenetic clock for age prediction using blood methylomes. 
Here we provide a commandline standalone python version of it.

This project is developed by Miri Varshavsky in [Prof. Tommy Kaplan's lab](https://www.cs.huji.ac.il/~tommy/) at the Hebrew University, Jerusalem, Israel.
 
## Requirements
GP-age runs on python3, and the following packages are required:

* matplotlib (required for GPy)
* pandas
* scikit-learn
* GPy

These packages can be installed by running the following commands:
```
pip install matplotlib
pip install pandas
pip install scikit-learn
pip install gpy
```

## Installation
```
git clone https://github.com/mirivar/GP-age.git
cd GP-age
```

## Arguments
The stand-alone has several arguments.
* -x: A path to a csv containing the methylation data. Rows are CpG sites, columns are samples, and the first line is a header (see `meth_test.csv` for an example)
* -y: A path to a file containing the ages of the samples. Optional. Two formats are supported:
  * A csv file, with first column containing the sample IDs, and second columns containing the ages. First line is a header line (see `age_test.csv` for an example)
  * A txt file, with a column of ages of the samples listed in same order as in the methylation array. First line is a header line.
* -o: Output directory, optional. If not provided, results will be printed to stdout.
* -t: Add if wish to run GP-age on the predefined test set (demo).

## Execution
The standalone should be executed by running the following command:
```
/cs/cbio/miri/thesis/epigenetic_clock/src/GP-age/predict_age.py <arguments>
```
The arguments provided will define the behavior of the prediction, as listed below.

### Demo
For a quick age prediction over a pre-defined test set, run:
```
/cs/cbio/miri/thesis/epigenetic_clock/src/GP-age/predict_age.py -t
```
The predictions and statistics will be printed to stdout.

### Full usage example
```
/cs/cbio/miri/thesis/epigenetic_clock/src/GP-age/predict_age.py -x <meth. array path> [-y <ages path> -o <output dir>]
```
Age of samples from the provided methylation array will be predicted. If an output dir is provided, they will be saved as a csv under `<output_dir>/GP-age_predictions.csv`. If no output dir is provided, predictions will be printed to stdout.

If an ages file is provided, statistics (RMSE, MedAE (median absolute error), and MeanAE (mean absolute error)) will be calculated. If an output dir is provided, they will be saved as a csv under `<output_dir>/GP-age_stats.csv`. If no output dir is provided, statistics will be printed to stdout.
