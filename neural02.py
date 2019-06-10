import pandas
import numpy
from sklearn.neural_network import MLPRegressor

FEATURES = [
    'Powierzchnia w m2',
    'Liczba pięter w budynku',
    'Liczba pokoi', #'Miejsce parkingowe'
    'Piętro',
    'Rok budowy'
]

def preprocess(data):
    #data = data.replace({'parter': 0, 'poddasze': 0, 'brak miejsca parkingowego' :0, 'przynależne na ulicy':1, 'pod wiatą':2, 'w garażu':3, 'parking strzeżony':4}, regex=True)
    data = data.replace({'parter': 0, 'poddasze': 0}, regex=True)
    data = data.applymap(numpy.nan_to_num) 
    return data

train_data = pandas.read_csv('train/train.tsv', header=0, sep='\t')
columns=train_data.columns[1:]  
X_train=train_data[FEATURES] 
X_train=preprocess(X_train)
Y_train=train_data['cena']

X_test= pandas.read_csv('test-A/in.tsv', header=None, sep='\t', names=columns)
X_test=X_test[FEATURES]
X_test=preprocess(X_test)

X_dev= pandas.read_csv('dev-0/in.tsv', header=None, sep='\t', names=columns)
X_dev=X_dev[FEATURES]
X_dev=preprocess(X_dev)

model=MLPRegressor(solver='lbfgs', alpha=0.001)
model.fit(X_train,Y_train)

predicted_ydev = model.predict(X_dev)
#print(predicted_ydev[0:20])
pandas.DataFrame(predicted_ydev).to_csv('dev-0/out.tsv', index=None, header=None, sep='\t')

predicted_ytest = model.predict(X_test)
#print(predicted_ydev[0:20])
pandas.DataFrame(predicted_ytest).to_csv('test-A/out.tsv', index=None, header=None, sep='\t')

