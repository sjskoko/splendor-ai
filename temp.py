from keras.models import load_model, model_from_json
from keras.models import Sequential
from keras.layers import Dense
import os
# if os.path.exists('/root/home/splendor/splendor-ai/splendor_ai/brains/model.h5'):
#    print(True)
# else:
#    print('not exist')

loaded_model = Sequential()

input_nodes = 1312

output_nodes = 88
# output_nodes = 13

hidden_layer_size = int((input_nodes+output_nodes) / 2)
print(hidden_layer_size)
# Input Layer
#loaded_model.add(InputLayer(batch_input_shape=(1, input_nodes)))
loaded_model.add(Dense(input_nodes, activation='linear', input_dim=input_nodes))

# Hidden layers
loaded_model.add(Dense(input_nodes, activation='sigmoid'))
#loaded_model.add(LeakyReLU(alpha=.001))
loaded_model.add(Dense(hidden_layer_size, activation='sigmoid'))
loaded_model.add(Dense(hidden_layer_size, activation='sigmoid'))

# Output layer
loaded_model.add(Dense(output_nodes, activation='linear'))

# print(loaded_model.weights)

# load weights and compile model
loaded_model.load_weights('/root/home/splendor/splendor-ai/splendor_ai/brains/model.h5')
loaded_model.compile(loss='mse', optimizer='adam', metrics=['mae'])




# # keras.models를 통해 만들어진 Sequential이나 Model



print(loaded_model.weights)