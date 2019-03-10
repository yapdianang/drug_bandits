import pandas as pd
import numpy as np

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

def get_data(path, seed=234):
	df = pd.read_csv(path)
	df = df.dropna(subset = ['Therapeutic Dose of Warfarin'])
	df['dosage_bucket'] = (df['Therapeutic Dose of Warfarin'] / 7).apply(get_bucket)
	df['Carbamazepine (Tegretol)'] = df['Carbamazepine (Tegretol)'].astype('object')
	df['Phenytoin (Dilantin)'] = df['Phenytoin (Dilantin)'].astype('object')
	df['Rifampin or Rifampicin'] = df['Rifampin or Rifampicin'].astype('object')

	feature_list = ['Gender', 'Race', 'Ethnicity', 'Age', 'Cyp2C9 genotypes', \
            'VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T', \
            'VKORC1 genotype: 497T>G (5808); chr16:31013055; rs2884737; A/C', \
            'VKORC1 QC genotype: 1173 C>T(6484); chr16:31012379; rs9934438; A/G', \
            'VKORC1 genotype: 1542G>C (6853); chr16:31012010; rs8050894; C/G', \
            'VKORC1 genotype: 3730 G>A (9041); chr16:31009822; rs7294;  A/G', \
            'VKORC1 genotype: 2255C>T (7566); chr16:31011297; rs2359612; A/G', \
            'VKORC1 genotype: -4451 C>A (861); Chr16:31018002; rs17880887; A/C', \
            ]
	# feature_list = ['Gender', 'Cyp2C9 genotypes', 
 #            'VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T', \
 #            'VKORC1 genotype: 497T>G (5808); chr16:31013055; rs2884737; A/C', \
 #            'VKORC1 QC genotype: 1173 C>T(6484); chr16:31012379; rs9934438; A/G', \
 #            'VKORC1 genotype: 1542G>C (6853); chr16:31012010; rs8050894; C/G', \
 #            'VKORC1 genotype: 3730 G>A (9041); chr16:31009822; rs7294;  A/G', \
 #            'VKORC1 genotype: 2255C>T (7566); chr16:31011297; rs2359612; A/G', \
 #            'VKORC1 genotype: -4451 C>A (861); Chr16:31018002; rs17880887; A/C']
	# feature_list = ['Gender']
	features = pd.get_dummies(df[feature_list], dummy_na=True)
	features['bias'] = 1
	features['dosage_bucket'] = df['dosage_bucket']
	features_array = np.array(features)
	np.random.seed(seed)
	np.random.shuffle(features_array)	
	# returns np array of features and ground truth
	return features_array[:,:-1], features_array[:,-1]

