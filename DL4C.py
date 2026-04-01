# Using a deep feed forward network with two hidden layers for performing linear regression and predicting values

from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler
x, y = make_regression(n_samples=100, n_features=2, noise=0.1, random_state=1)
scalarx, scalary = MinMaxScaler(), MinMaxScaler()
x = scalarx.fit_transform(x)
y = scalary.fit_transform(y.reshape(-1, 1))
model = Sequential()

#model.add(Dense(4, input_dim=2, activation='relu'))
model.add(Dense(4, input_shape=(2,), activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=500, batch_size=10, verbose=1)
xnew_generated_features, _ = make_regression(n_samples=3, n_features=2, noise=0.1, random_state=1)
xnew = scalarx.transform(xnew_generated_features)
ynew = model.predict(xnew)
for i in range(len(xnew)):
  print("X=%s, Predicted (scaled)=%s" % (xnew[i], ynew[i]))
y_original = scalary.inverse_transform(ynew)
print("Actual Predicted Values:", y_original)
