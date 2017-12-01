from pickle import dump
from pickle import load
from numpy import reshape
from numpy import asarray
from numpy import transpose
from sklearn.decomposition import PCA

# load original features
features = load(open('features.pkl', 'rb'))

# record array values and list keys
list_values = [v for v in features.values()]
list_keys = [k for k in features]
array_values = asarray(list_values)

# reduce dimentions using PCA
values = array_values.reshape(array_values.shape[0],-1)
pca = PCA(n_components=800)
fit = pca.fit(transpose(values))
new_values = transpose(fit.components_)
new_array_values = new_values.reshape(new_values.shape[0],1,-1)

# produce reduced features
new_features = dict()
for j in range(len(list_keys)):
    new_features[list_keys[j]] = new_array_values[j]

dump(new_features, open('reduced_800_features.pkl', 'wb'))
