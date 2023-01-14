# Dota2-Prediction-Model

Files:
- runModel.py has a terminal interface. Run this program directly, python runModel.py,
and follow the instruction to select dataset and model. Note that most of them take a while,
up to several hours especially when you draw learning curve.

- preprocessing.py is a library of functions we used to preprocess the data,
such as binarize, adding PolynomialFeatures, and adding synergy and counter.

- files ended .npz are the dataset. You can also regenerate these file using
functions in preprocessing.py, but since most of them take a long time and we
reused a lot, they are included here. For the same reason, trainLabel.npy and
testLabel.npy are included here.

Folders:
- load_data contains the original dataset, and addHeader.py helps transform the
dataset into a format that can be opened as arff in weka.

- newData contains the data we grabbed using datdota. parser.py turns the data
into a format acceptable for preprocessing.py and runModel.py. heroes.json is
the mapping of hero names to hero id.

- plot contains some plots of learning curve.

