# GP-age

GP-age is an epigenetic clock for age prediction using blood methylomes, based on a cohort-based non-parametric Gaussian Process regression model. 

Here we provide a commandline standalone python version of it. We provide three models which use 10, 30 and 80 CpGs, and additional three models (termed a, b, c), which use disjoint groups of CpGs and are therefore considered independent. It shall be noted that the full 10-, 30-, or 80-CpG models perform better than the independent models, but the latter may provide biological insights when inspecting coordinated deviations of predicted age from chronological age.

This project is developed by Miri Varshavsky in [Prof. Tommy Kaplan's lab](https://www.cs.huji.ac.il/~tommy/) at the Hebrew University, Jerusalem, Israel.
 
## Requirements
GP-age runs on python3, and the following packages are required:

* matplotlib (required for GPy, tested on version 3.4.1)
* pandas (tested on version 1.3.5)
* scikit-learn (tested on version 0.24.2)
* GPy (tested on version 1.10.0)

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
Installation should take less than 1 minute.

## Arguments
The stand-alone has several arguments.
* -x: A path to a csv containing the methylation data. Rows are CpG sites, columns are samples, and the first line is a header (see `meth_test.csv` for an example)
* -y: A path to a file containing the ages of the samples. Optional. Two formats are supported:
  * A csv file, with first column containing the sample IDs, and second columns containing the ages. First line is a header line (see `age_test.csv` for an example)
  * A txt file, with a column of ages of the samples listed in same order as in the methylation array. First line is a header line.
* -m: Model type to run. Can be either the model size (10, 30 or 80) or the model from the three independent models (a, b, c). Default is 30.
* -o: Output directory, optional. If not provided, results will be printed to stdout.
* -t: Add if wish to run GP-age on the predefined test set (demo).

## Execution
The standalone should be executed by running the following command:
```
predict_age.py <arguments>
```
The arguments provided will define the behavior of the prediction, as listed below.

### Demo
For a quick age prediction over a pre-defined test set, run:
```
predict_age.py -t
```
The predictions and statistics will be printed to stdout. Running the demo should take less than 2 minutes. An example of the output can be found in the demo_results folder.

### Full usage example
```
predict_age.py -x <meth. array path> [-y <ages path> -o <output dir> -m <model type>]
```
Age of samples from the provided methylation array will be predicted using a GP-age model of type m (default is m=30). If an output dir is provided, they will be saved as a csv under `<output_dir>/GP-age_<m>_cpgs_predictions.csv` if m is the models size, or `<output_dir>/GP-age_<m>_predictions.csv` if m is an independent model. If no output dir is provided, predictions will be printed to stdout.

If an ages file is provided, statistics (RMSE, MedAE (median absolute error), and MeanAE (mean absolute error)) will be calculated. If an output dir is provided, they will be saved as a csv under `<output_dir>/GP-age_<m>_cpgs_stats.csv` if m is the models size, or `<output_dir>/GP-age_<m>_stats.csv` if m is an independent model. If no output dir is provided, statistics will be printed to stdout.
