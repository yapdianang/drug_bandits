import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class DataStream(object):
	# Read in a csv, shuffle rows.
	# Iterator below.
	# For each row:
		# yield feature vector extracted from that row, and ground truth action

	def __init__(self, csv_path, val=True, seed=234):
		# make train and test the same if not running on validation set
		self.table, y = get_data(csv_path, seed)
		self.table_test = self.table
		self.ground_truth, self.dosage = y[:,0], y[:,1]
		self.ground_truth_test, self.dosage_test = y[:,0], y[:,1]

		# if validation set, then split the data
		if val:
			self.table, self.table_test, y, y_test = \
					train_test_split(self.table, y, test_size=0.1, random_state=seed, stratify=y[:,0])
			self.ground_truth, self.dosage = y[:,0], y[:,1]
			self.ground_truth_test, self.dosage_test = y_test[:,0], y_test[:,1]

		self.max_rows = len(self.table)
		self.feature_dim = self.table.shape[-1]
		self.current = 0

	# Iterator methods
	def __iter__(self):
		return self

	def __next__(self):
		if self.current >= self.max_rows:
			raise StopIteration
		else:
			output = (self.table[self.current], self.ground_truth[self.current], self.dosage[self.current]) 
			self.current += 1
			return output

	def __len__(self):
		return len(self.table)


def get_bucket(dosage):
	if dosage < 3:
		return 'low'
	elif dosage <= 7:
		return 'medium'
	else:
		return 'high'

def get_arm_from_bucket_name(lowmedhigh):
	if lowmedhigh == "low":
		return 1
	elif lowmedhigh == "medium":
		return 2
	elif lowmedhigh == "high":
		return 3
	else:
		raise ValueError('you don goofed')


def bucket_weight(x):
	return str((x-30)//20)
	# return str(x//50)


def bucket_height(x):
	return str((x-120)//10)
	# return str(x//50)


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

# normalize to (0,1) range
def normalize_col(col):
	mx, mn =  np.max(col), np.min(col)
	return (col - mn) / (mx-mn)

def get_data(path, seed=234):
	df = pd.read_csv(path)
	df = df.dropna(subset = ['Therapeutic Dose of Warfarin'])
	df['dosage_bucket'] = (df['Therapeutic Dose of Warfarin'] / 7).apply(get_bucket)
	df['weight_bucket'] = df['Weight (kg)'].apply(bucket_weight)
	df['height_bucket'] = df['Height (cm)'].apply(bucket_height)

	feature_list = ['Gender', 'Race', 'Ethnicity', 'Age', 'Cyp2C9 genotypes', \
			# 'weight_bucket', 'height_bucket', 
			# 'Aspirin', 'Acetaminophen or Paracetamol (Tylenol)',
			'Was Dose of Acetaminophen or Paracetamol (Tylenol) >1300mg/day',
			'Simvastatin (Zocor)', 'Atorvastatin (Lipitor)', 'Fluvastatin (Lescol)',
			'Lovastatin (Mevacor)', 'Pravastatin (Pravachol)',
			'Rosuvastatin (Crestor)', 'Cerivastatin (Baycol)',
			'Amiodarone (Cordarone)', 'Carbamazepine (Tegretol)',
			'Phenytoin (Dilantin)', 'Rifampin or Rifampicin',
			'Sulfonamide Antibiotics', 'Macrolide Antibiotics',
			'Anti-fungal Azoles', 'Herbal Medications, Vitamins, Supplements',
			'VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T', \
			'VKORC1 genotype: 497T>G (5808); chr16:31013055; rs2884737; A/C', \
			'VKORC1 QC genotype: 1173 C>T(6484); chr16:31012379; rs9934438; A/G', \
			'VKORC1 genotype: 1542G>C (6853); chr16:31012010; rs8050894; C/G', \
			'VKORC1 genotype: 3730 G>A (9041); chr16:31009822; rs7294;  A/G', \
			'VKORC1 genotype: 2255C>T (7566); chr16:31011297; rs2359612; A/G', \
			'VKORC1 genotype: -4451 C>A (861); Chr16:31018002; rs17880887; A/C', \
			'Estimated Target INR Range Based on Indication','Indication for Warfarin Treatment', 
			'Subject Reached Stable Dose of Warfarin', 'Current Smoker']

	# feature_list = ['Gender']
	# feature_list = ['weight_bucket', 'height_bucket', 'Gender', 'Race', 'Ethnicity', 'Age', 'Cyp2C9 genotypes', \
 #            'VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T', \
 #            'VKORC1 genotype: 497T>G (5808); chr16:31013055; rs2884737; A/C', \
 #            'VKORC1 QC genotype: 1173 C>T(6484); chr16:31012379; rs9934438; A/G', \
 #            'VKORC1 genotype: 1542G>C (6853); chr16:31012010; rs8050894; C/G', \
 #            'VKORC1 genotype: 3730 G>A (9041); chr16:31009822; rs7294;  A/G', \
 #            'VKORC1 genotype: 2255C>T (7566); chr16:31011297; rs2359612; A/G', \
 #            'VKORC1 genotype: -4451 C>A (861); Chr16:31018002; rs17880887; A/C', \
 #            'Indication for Warfarin Treatment', 'Target INR']

	# cast to object so we can get dummies (1 hots)
	df[feature_list] = df[feature_list].astype('object')

	features = pd.get_dummies(df[feature_list], dummy_na=True)
	features['height'] = df['Height (cm)'].fillna(np.mean(df['Height (cm)']))
	features['weight'] = df['Weight (kg)'].fillna(np.mean(df['Weight (kg)']))
	features['Target INR'] = df['Target INR'].fillna(np.mean(df['Target INR']))
	features['bias'] = 1

	features['dosage_bucket'] = df['dosage_bucket']
	features['real_dosage'] = df['Therapeutic Dose of Warfarin'] / 7

	features_array = np.array(features)
	np.random.seed(seed)
	np.random.shuffle(features_array)	
	# returns np array of features and ground truth bucket and real valued dosage
	return features_array[:,:-2], features_array[:,-2:]

