import GPy
import argparse

import pandas as pd
from sklearn.metrics import mean_squared_error, median_absolute_error, mean_absolute_error
from sklearn.impute import SimpleImputer


def calc_stats(prediction, y):
	rmse = mean_squared_error(y, prediction, squared=False)
	med_ae = median_absolute_error(y, prediction)
	mean_ae = mean_absolute_error(y, prediction)

	return pd.DataFrame({'stats': [rmse, med_ae, mean_ae]}, index=['RMSE', 'MedAE', 'MeanAE'])


def predict(X_path, y_path=None):
	sites = pd.read_csv('blood_dynamical_range_adults_0.2_high_quality_no_child_GSE_corr_0.4.csv').iloc[:, 0]
	X = pd.read_csv(X_path, index_col=0)
	X = X[sites]

	predictor = GPy.models.GPRegression.load_model('blood_dynamical_range_adults_0.2_high_quality_no_child_GSE_corr_0.4.json.zip')
	if X.isna().any(axis=None):
		imputer = SimpleImputer().fit(pd.DataFrame(predictor.X, columns=sites))
		X = pd.DataFrame(data=imputer.transform(X), index=X.index, columns=X.columns.values.tolist())

	predictions = predictor.predict(X.values)[0]

	if y_path is not None:
		y = pd.read_csv(y_path, index_col=0).iloc[:, 0]
		stats = calc_stats(predictions, y)


	a=1


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('-x', help='csv path of methylation array', required=False)
	parser.add_argument('-y', help='csv path to real age', required=False)
	parser.add_argument('-o', help='output directory')
	parser.add_argument('-t', '--test', action='store_true', help='run test on provided test set')

	args = parser.parse_args()

	assert args.x is not None or args.test
	if args.test:
		x = 'x_test.csv'
		y = 'y_test.csv'

	else:
		x = args.x
		y = args.y

	predict(X_path=x, y_path=y)