#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import pandas as pd
import ast 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Concatenate, Reshape , Flatten
from tensorflow.keras.layers import Reshape
from keras.layers import Input, LSTM, Dense, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from keras.layers import TimeDistributed
from tensorflow.keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import ast
import h5py


# In[12]:


df = pd.read_csv('text_num_emebddings.csv')


# In[13]:


# Retrieving data from the HDF5 file
with h5py.File('text_embeddings.h5', 'r') as hf:
    # Retrieve title_embeddings
    title_embedding = np.array(hf['title_embedding'])

    # Retrieve description_embeddings
    video_description_embedding = np.array(hf['video_description_embedding'])
                                          
    # Retrieve channel_name_embedding
    channel_name_embedding = np.array(hf['channel_name_embedding'])
    
    # Retrieve comments_embedding
    comments_embedding = np.array(hf['comments_embedding'])


# In[14]:


# Concatenate the text embeddings horizontally
X_text_combined = np.concatenate((title_embedding, video_description_embedding, channel_name_embedding, comments_embedding), axis=1)

# Combine the numeric features and text embeddings
X_combined = np.concatenate((X_text_combined, df[['subscriber_count', 'num_likes', 'view_count', 'num_comments']].values), axis=1)


# In[15]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_combined,
    df['deceptive_flag'].values,
    test_size=0.2,
    random_state=42
)


# In[17]:


# Build the model
model = Sequential()
model.add(Dense(units=128, activation='relu', input_dim=X_combined.shape[1]))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Evaluate the model
accuracy = model.evaluate(X_test, y_test)[1]
print(f"Test Accuracy: {accuracy}")


# In[19]:


import matplotlib.pyplot as plt

# Train the model and obtain the history
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[22]:


# Sample a few examples from the training set
sample_indices_train = np.random.choice(X_train.shape[0], size=100, replace=False)
X_sample_train = X_train[sample_indices_train]
y_sample_train = y_train[sample_indices_train]

# Sample a few examples from the test set
sample_indices_test = np.random.choice(X_test.shape[0], size=50, replace=False)
X_sample_test = X_test[sample_indices_test]
y_sample_test = y_test[sample_indices_test]

# Predictions on the training set
predictions_train = model.predict(X_sample_train)

# Predictions on the test set
predictions_test = model.predict(X_sample_test)

# Plot predictions for the training set
plt.scatter(range(len(y_sample_train)), y_sample_train, label='Actual', marker='o')
plt.scatter(range(len(predictions_train)), predictions_train, label='Predicted', marker='x')
plt.title('Actual vs Predicted Labels (Training Set)')
plt.xlabel('Sample Index')
plt.ylabel('Binary Label')
plt.legend()
plt.show()

# Plot predictions for the test set
plt.scatter(range(len(y_sample_test)), y_sample_test, label='Actual', marker='o')
plt.scatter(range(len(predictions_test)), predictions_test, label='Predicted', marker='x')
plt.title('Actual vs Predicted Labels (Test Set)')
plt.xlabel('Sample Index')
plt.ylabel('Binary Label')
plt.legend()
plt.show()


# In[23]:


model.save('deception_model_dense')


# In[ ]:




