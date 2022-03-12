import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.metrics import mean_absolute_error
from tensorflow.keras.metrics import mean_squared_error
from tensorflow.keras.metrics import mean_absolute_percentage_error

df = pd.read_csv("traffic.csv")
df

junction_1_data = df[df['Junction']==1]

len(junction_1_data)

vehicles_1 = junction_1_data['Vehicles'].to_numpy()
vehicles_1 = vehicles_1[-3000:]
len(vehicles_1)


WINDOW_SIZE = 7
HORIZON = 1


def create_data(x,horizon=HORIZON,window_size=WINDOW_SIZE):
    def get_labelled_windows(x,horizon=HORIZON):
        """
        creating data for time series forecasting problem
        """
        return x[:,:-horizon],x[:,-horizon:]
    
    
    def make_windows(x,window_size=WINDOW_SIZE,horizon=HORIZON):
        window_step = np.expand_dims(np.arange(window_size+horizon),axis=0)
        window_indexes =window_step + np.expand_dims(np.arange(len(x)-(window_size+horizon-1)),axis=0).T
        windowed_array = x[window_indexes]
        windows,labels = get_labelled_windows(windowed_array,horizon=horizon)
        return windows,labels
    full_windows,full_labels = make_windows(x,window_size=WINDOW_SIZE,horizon=HORIZON)
    return full_windows,full_labels



windows_j1,labels_j1 = create_data(vehicles_1,horizon=HORIZON,window_size=WINDOW_SIZE)

windows_j1.shape,labels_j1.shape

def train_test_split(windows,labels,test_size=0.2):
    split_size = int(len(windows)*(1-test_size))
    train_windows = windows[:split_size]
    train_labels = labels[:split_size]
    test_windows = windows[split_size:]
    test_labels = labels[split_size:]
    return train_windows,train_labels,test_windows,test_labels

train_windows_j1,train_labels_j1,test_windows_j1,test_labels_j1 = train_test_split(windows_j1,labels_j1)



data_nbeats = junction_1_data.copy()
data_nbeats = data_nbeats.drop(["Junction","ID"],axis=1)
data_nbeats.set_index("DateTime")
for i in range(WINDOW_SIZE):
  data_nbeats[f"Vehicles+{i+1}"] = data_nbeats["Vehicles"].shift(periods=i+1)
data_nbeats.dropna().head()

data_nbeats.index = data_nbeats["DateTime"]
data_nbeats = data_nbeats.drop(data_nbeats.columns[0],axis=1)

X = data_nbeats.dropna().drop("Vehicles", axis=1)
y = data_nbeats.dropna()["Vehicles"]

# Make train and test sets
split_size = int(len(X) * 0.8)
X_train, y_train = X[:split_size], y[:split_size]
X_test, y_test = X[split_size:], y[split_size:]
len(X_train), len(y_train), len(X_test), len(y_test)

# 1. Turn train and test arrays into tensor Datasets
train_features_dataset = tf.data.Dataset.from_tensor_slices(X_train)
train_labels_dataset = tf.data.Dataset.from_tensor_slices(y_train)

test_features_dataset = tf.data.Dataset.from_tensor_slices(X_test)
test_labels_dataset = tf.data.Dataset.from_tensor_slices(y_test)

# 2. Combine features & labels
train_dataset = tf.data.Dataset.zip((train_features_dataset, train_labels_dataset))
test_dataset = tf.data.Dataset.zip((test_features_dataset, test_labels_dataset))

# 3. Batch and prefetch for optimal performance
BATCH_SIZE = 1024 # taken from Appendix D in N-BEATS paper
train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

train_dataset, test_dataset

from tensorflow.keras import layers

# Create NBeatsBlock custom layer 
class NBeatsBlock(tf.keras.layers.Layer):
  def __init__(self, # the constructor takes all the hyperparameters for the layer
               input_size: int,
               theta_size: int,
               horizon: int,
               n_neurons: int,
               n_layers: int,
               **kwargs): # the **kwargs argument takes care of all of the arguments for the parent class (input_shape, trainable, name)
    super().__init__(**kwargs)
    self.input_size = input_size
    self.theta_size = theta_size
    self.horizon = horizon
    self.n_neurons = n_neurons
    self.n_layers = n_layers

    # Block contains stack of 4 fully connected layers each has ReLU activation
    self.hidden = [tf.keras.layers.Dense(n_neurons, activation="relu") for _ in range(n_layers)]
    # Output of block is a theta layer with linear activation
    self.theta_layer = tf.keras.layers.Dense(theta_size, activation="linear", name="theta")

  def call(self, inputs): # the call method is what runs when the layer is called 
    x = inputs 
    for layer in self.hidden: # pass inputs through each hidden layer 
      x = layer(x)
    theta = self.theta_layer(x) 
    # Output the backcast and forecast from theta
    backcast, forecast = theta[:, :self.input_size], theta[:, -self.horizon:]
    return backcast, forecast

N_EPOCHS = 5000 # called "Iterations" in Table 18
N_NEURONS = 512 # called "Width" in Table 18
N_LAYERS = 4
N_STACKS = 30

INPUT_SIZE = WINDOW_SIZE*HORIZON # called "Lookback" in Table 18
THETA_SIZE = INPUT_SIZE + HORIZON


tf.random.set_seed(42)
from tensorflow.keras import layers

# 1. Setup N-BEATS Block layer
nbeats_block_layer = NBeatsBlock(input_size=INPUT_SIZE,
                                 theta_size=THETA_SIZE,
                                 horizon=HORIZON,
                                 n_neurons=N_NEURONS,
                                 n_layers=N_LAYERS,
                                 name="InitialBlock")

# 2. Create input to stacks
stack_input = layers.Input(shape=(INPUT_SIZE), name="stack_input")

# 3. Create initial backcast and forecast input (backwards predictions are referred to as residuals in the paper)
backcast, forecast = nbeats_block_layer(stack_input)
# Add in subtraction residual link, thank you to: https://github.com/mrdbourke/tensorflow-deep-learning/discussions/174 
residuals = layers.subtract([stack_input, backcast], name=f"subtract_00") 

# 4. Create stacks of blocks
for i, _ in enumerate(range(N_STACKS-1)): # first stack is already creted in (3)

  # 5. Use the NBeatsBlock to calculate the backcast as well as block forecast
  backcast, block_forecast = NBeatsBlock(
      input_size=INPUT_SIZE,
      theta_size=THETA_SIZE,
      horizon=HORIZON,
      n_neurons=N_NEURONS,
      n_layers=N_LAYERS,
      name=f"NBeatsBlock_{i}"
  )(residuals) # pass it in residuals (the backcast)

  # 6. Create the double residual stacking
  residuals = layers.subtract([residuals, backcast], name=f"subtract_{i}") 
  forecast = layers.add([forecast, block_forecast], name=f"add_{i}")

# 7. Put the stack model together
model_7 = tf.keras.Model(inputs=stack_input, 
                         outputs=forecast, 
                         name="model_7_N-BEATS")
# 8. Compile with MAE loss and Adam optimizer
model_7.compile(loss="mae",
                optimizer=tf.keras.optimizers.Adam(0.001),
                metrics=["mae", "mse","accuracy"])

# 9. Fit the model with EarlyStopping and ReduceLROnPlateau callbacks
model_7.fit(train_dataset,
            test_dataset,
            epochs=500,
            # callbacks=[create_model_checkpoint(model_name=stack_model.name)] # saving model every epoch consumes far too much time
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=200, restore_best_weights=True),
                      tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=100, verbose=1)])

# Evaluate N-BEATS model on the test dataset
model_7.evaluate(test_dataset)

nbeats_preds = model_7.predict(test_dataset)

import matplotlib.pyplot as plt
plt.plot(tf.range(24),test_labels_j1[0],label="ground truth value")
plt.plot(tf.range(24),nbeats_preds[0],label="N Beats predictions")
plt.legend()
plt.figure(figsize=(10,7))
plt.plot(tf.range(24),test_labels_j1[1],label="ground truth value")
plt.plot(tf.range(24),nbeats_preds[1],label="N Beats predictions")
plt.legend()

model_7.save("best_model_NBeats")

