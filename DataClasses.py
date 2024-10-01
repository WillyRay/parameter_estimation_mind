import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.model_selection import train_test_split

from dataclasses import dataclass


@dataclass
class Run:
    id: int
    decay: float
    touchTransferFraction: float
    counts: list[int]
    occupancies: list[int]
    cdffs: list[int]
    anyCps: list[int]

@dataclass
class Sample:
        run: int
        startDay: int
        decay: float
        touchTransferFractions: float
        counts: list[int]
        occupancies: list[int]
        cdiffs: list[int]
        anyCps: list[int]





def split_sequences(sequences, n_steps):
    retlist=list()
    for i in range(len(sequences)):
        end_ix = i + n_steps
        if end_ix > len(sequences):
            break
        seq_x= sequences[i:end_ix]
        retlist.append(seq_x)

    return np.array(retlist)

def get_samples(run, n_steps):
    for seq in [run.counts, run.cdffs, run.occupancies, run.anyCps]:
        pass


data = pd.read_csv("./data/sim_data.csv", index_col=["run"]).sort_index()

counts = []
cdffs = []
occupancies = []
anyCps = []
decays = []
touchTransferFractions = []


runs = []
for run in data.index.unique():
    r = Run(id=run, touchTransferFraction=data.loc[run]["surfaceTransferFraction"][0:1].values[0], decay=data.loc[run]["decayRate"][0:1].values[0], counts=data.loc[run]["count"], cdffs=data.loc[run]["CDIFF"], occupancies=data.loc[run]["occupancy"], anyCps=data.loc[run]["anyCP"])
    runs.append(r)

for r in runs:
    pass

sys.exit(0)





for run in data.index.unique():
    decays.append(data.loc[run]["decayRate"][0:1].values[0])
    touchTransferFractions.append(data.loc[run]["surfaceTransferFraction"][0:1].values[0])
    counts.append([data.loc[run]["count"]])
    cdffs.append([data.loc[run]["CDIFF"]])
    occupancies.append([data.loc[run]["occupancy"]])
    anyCps.append([data.loc[run]["anyCP"]])




X = np.stack((np.array(counts), np.array(cdffs), np.array(occupancies), np.array(anyCps)), axis=1)
y = np.stack((np.array(decays), np.array(touchTransferFractions)), axis=1)





counts = np.array(counts)
cdffs = np.array(cdffs)
occupancies = np.array(occupancies)
anyCps = np.array(anyCps)
decays = np.array(decays)
touchTransferFractions = np.array(touchTransferFractions)

counts = counts.reshape((len(counts), 1))
cdffs = cdffs.reshape((len(cdffs), 1))
occupancies = occupancies.reshape((len(occupancies), 1))
anyCps = anyCps.reshape((len(anyCps), 1))
decays = decays.reshape((len(decays), 1))
touchTransferFractions = touchTransferFractions.reshape((len(touchTransferFractions), 1))

dataset = np.hstack((counts, cdffs, occupancies, anyCps))
out_seq = np.hstack((decays, touchTransferFractions))

nsteps = 56

X = split_sequences(dataset, nsteps)
print(X)

sys.exit(0)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train.reshape(-1, 276, 4)
X_test = X_test.reshape(-1, 276, 4)

model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(276,4)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(2))

model.compile(optimizer='adam', loss='mse')

model.fit(X_train, y_train, epochs=10, verbose=0)

loss = model.evaluate(X_test, y_test)
print("Test Loss:", loss)

predictions = model.predict(X_test)
print("Predicted Parameters:", predictions)

print(X_test)
print(y_test)
