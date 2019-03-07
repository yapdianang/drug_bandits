import pandas as pd

def get_bucket(dosage):
	if dosage < 3:
		return 'low'
	elif dosage <= 7:
		return 'medium'
	else:
		return 'high'

def get_accuracy(pred, real):
	x = pred == real
	return sum(x) / len(x)

def get_features_and_dosage(file):
	df = pd.read_csv(file)    
	feature_list = ['Bias', 'Age in decades', 'Height (cm)', 'Weight (kg)', 'Asian', 'Black or African American',
					'Missing or Mixed Race','Enzyme inducer status', 'Amiodarone status']
	features = df[feature_list]
	true_dosages = df['dosage_bucket']

	return df, features, true_dosages