#!/usr/bin/env python3
import os
import GPy
import argparse
import logging
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, median_absolute_error, mean_absolute_error
from sklearn.impute import SimpleImputer

os.getcwd()
logging.basicConfig()
logger = logging.getLogger('GP-age')
logger.setLevel(logging.INFO)


def calc_stats(prediction, y):
	rmse = mean_squared_error(y, prediction, squared=False)
	med_ae = median_absolute_error(y, prediction)
	mean_ae = mean_absolute_error(y, prediction)

	return pd.DataFrame({'stats': [rmse, med_ae, mean_ae]}, index=['RMSE', 'MedAE', 'MeanAE'])


def output_results(results, results_type, output_dir, title):
	if output_dir is not None:
		output_path = os.path.join(output_dir, f'GP-age_{title}_{results_type}.csv')
		results.to_csv(output_path, float_format='%.3f')
		logger.info(f'{results_type.capitalize()} were saved to {output_path}')

	else:
		print('\n', results.round(3), '\n')


def predict(m, X_path, y_path=None, output_dir=None):
	title = f'{m}_cpgs' if m not in ['a', 'b', 'c'] else m
	logger.info(f'Starting age prediction using GP-age ({title})')

	# Load methylation data for pre-defined CpG sites
	sites = pd.read_csv(f'model_data/GP-age_sites_{title}.csv').iloc[:, 0]
	logger.info(f'Loading methylation data...')
	X = pd.read_csv(X_path, index_col=0).T
	y = None
	if y_path is None and 'age' in X.columns:
		y = X['age']

	methylation_array = pd.DataFrame(np.nan, index=X.index, columns=sites)
	existing_sites = pd.Series(np.intersect1d(sites.values, X.columns.values))
	methylation_array[existing_sites] = X[existing_sites]
	X = methylation_array
	logger.info(f'Successfully loaded {X.shape[0]} samples')

	# Load GP-age model
	logger.info(f'Loading GP-age model...')
	predictor = GPy.models.GPRegression.load_model(f'model_data/GP-age_model_{title}.json.zip')
	logger.info('GP-age successfully loaded')

	# Fill missing values with the mean beta value of the specific CpG across train samples
	if X.isna().any(axis=None):
		logger.info(f'Imputing {X.isna().values.sum()} missing values')
		imputer = SimpleImputer().fit(pd.DataFrame(predictor.X, columns=sites))
		X = pd.DataFrame(data=imputer.transform(X), index=X.index, columns=X.columns)

	# Predict age
	logger.info('Starting age prediction...')
	predictions = pd.DataFrame({'predictions': predictor.predict(X.values)[0].squeeze()}, index=X.index)
	predictions.index.name = 'sample'
	logger.info('Age prediction completed')
	output_results(predictions, 'predictions', output_dir, title)

	# If real age is provided, calculate prediction statistics
	if y_path is not None or y is not None:
		logger.info('Calculating statistics')

		if y is None:
			y = pd.read_csv(y_path)

			# For support of 2-column csv files
			if y.shape[1] > 1:
				y.index = y.iloc[:, 0]
				y = y.loc[X.index]
			y = y.iloc[:, -1]

		stats = calc_stats(predictions, y)
		output_results(stats, 'stats', output_dir, title)

	logger.info('Done!')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('-x', help='csv path of methylation array', required=False)
	parser.add_argument('-y', help='csv path to real age', required=False)
	parser.add_argument('-o', '--output', help='output directory. If None, will print results to stdout')
	parser.add_argument('-t', '--test', action='store_true', help='run test on provided test set')
	parser.add_argument('-m', help='model to run (10, 30, 71, a, b, or c)', choices=['10', '30', '71', 'a', 'b', 'c'], default='30')
	args = parser.parse_args()

	assert args.x is not None or args.test
	if args.test:
		x = 'demo/meth_test.csv'
		y = 'demo/age_test.csv'

	else:
		x = args.x
		y = args.y

	predict(m=args.m, X_path=x, y_path=y, output_dir=args.output)