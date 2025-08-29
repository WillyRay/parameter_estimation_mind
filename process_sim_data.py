import pandas as pd

# Load simulation data
sim_data = pd.read_csv('data/sim_data.csv')

# Select features and targets
features = ['occupancy', 'anyCP', 'CDIFF', 'tick']
targets = ['decayRate', 'surfaceTransferFraction']

# Prepare X (features) and y (targets)
X = sim_data[features]
y = sim_data[targets]

# Save processed data for model training
X.to_csv('data/sim_features.csv', index=False)
y.to_csv('data/sim_targets.csv', index=False)

print('Feature and target files created: sim_features.csv, sim_targets.csv')
