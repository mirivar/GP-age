#!/usr/bin/env python3

import GPy
import argparse

import pandas as pd

import os

from sklearn.metrics import mean_squared_error, median_absolute_error, mean_absolute_error
from sklearn.impute import SimpleImputer

os.getcwd()


def calc_stats(prediction, y):
	rmse = mean_squared_error(y, prediction, squared=False)
	med_ae = median_absolute_error(y, prediction)
	mean_ae = mean_absolute_error(y, prediction)

	return pd.DataFrame({'stats': [rmse, med_ae, mean_ae]}, index=['RMSE', 'MedAE', 'MeanAE'])


def predict(X_path, y_path=None, output_dir=None):
	sites = pd.read_csv('blood_dynamical_range_adults_0.2_high_quality_no_child_GSE_corr_0.4_sites.csv').iloc[:, 0]
	X = pd.read_csv(X_path, index_col=0).T
	X = X[sites]

	predictor = GPy.models.GPRegression.load_model('blood_dynamical_range_adults_0.2_high_quality_no_child_GSE_corr_0.4.json.zip')
	if X.isna().any(axis=None):
		imputer = SimpleImputer().fit(pd.DataFrame(predictor.X, columns=sites))
		X = pd.DataFrame(data=imputer.transform(X), index=X.index, columns=X.columns.values.tolist())

	predictions = pd.DataFrame({'predictions': predictor.predict(X.values)[0].squeeze()}, index=X.index)
	predictions.index.name = 'sample'

	if output_dir is not None:
		predictions.to_csv(os.path.join(output_dir, 'GP-age_predictions.csv'), float_format='%.3f')

	else:
		print('Predictions:\n', predictions)

	if y_path is not None:
		y = pd.read_csv(y_path)
		if y.shape[1] > 1:
			y.index = y.iloc[:, 0]
			y = y.loc[X.index]
		y = y.iloc[:, -1]

		stats = calc_stats(predictions, y)
		if output_dir is not None:
			stats.to_csv(os.path.join(output_dir, 'GP-age_stats.csv'), float_format='%.3f')
		else:
			print('\nstats:\n', stats)

	a=1


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('-x', help='csv path of methylation array', required=False)
	parser.add_argument('-y', help='csv path to real age', required=False)
	parser.add_argument('-o', '--output', help='output directory. If None, will print results to stdout')
	parser.add_argument('-t', '--test', action='store_true', help='run test on provided test set')

	args = parser.parse_args()

	assert args.x is not None or args.test
	if args.test:
		x = 'meth_test.csv'
		y = 'age_test.csv'

	else:
		x = args.x
		y = args.y

	predict(X_path=x, y_path=y, output_dir=args.output)
