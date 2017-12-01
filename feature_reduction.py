from pickle import load 
from pickle import dump
from sklearn.decomposition import PCA

# load a pre-defined list of photo identifiers
def load_set(filename):
    doc = load_doc(filename)
    dataset = list()
    # process line by line
    for line in doc.split('\n'):
        # skip empty lines
        if len(line) < 1:
            continue
        # get the image identifier
        identifier = line.split('.')[0]
        dataset.append(identifier)
    return set(dataset)

# load photo features
def load_photo_features(filename):
    # load all features
    all_features = load(open(filename, 'rb'))
    # filter features
    features = all_features
    return features

# photo features
features = load_photo_features('features.pkl')
# feature extraction
pca = PCA(n_components=3000)
fit = pca.fit(features)
features_reduced = fit.components_
# save to file
dump(features_reduced, open('features_pca.pkl','wb'))
