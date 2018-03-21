import pandas as pd

from topopy import MorseSmaleComplex

# df = pd.read_csv('simulations.csv')
# X = df[df.columns[0:5]].as_matrix()
# Y = df[df.columns[7]].as_matrix().flatten()

df = pd.read_csv('combustion_cleaned.csv')
X = df[df.columns[0:9]].as_matrix()
Y = df[df.columns[10]].as_matrix().flatten()

msc = MorseSmaleComplex(graph='beta skeleton', max_neighbors=500,
                        normalization='zscore', debug=True)

msc.build(X, Y)