import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
import numpy as np
import cv2
import re
import h5py

# Check if GPUs are available
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("No GPUs available, using CPU.")

# Import deceptive dataset
df_d = pd.read_csv("deceptive_vid.csv")
# Adding deceptive flag 
df_d['deceptive_flag'] = 0 
# Import non-deceptive dataset
df_nd = pd.read_csv("nondeceptive_vid.csv")
# Adding non-deceptive flag 
df_nd['deceptive_flag'] = 1

df = pd.concat([df_d, df_nd], ignore_index=True)
# Drop Category
df.drop(columns=['Category'], inplace=True)
# Remove if nan Video title
df = df[df['video_name'].notna()]

df['video_title'] = df['video_title'].astype(str)
df['channel_name'] = df['channel_name'].astype(str)
df['video_description'] = df['video_description'].astype(str)
df['comments'] = df['comments'].astype(str)

# Process only the first 350 rows
df = df.head(690)

# Assume you have a pre-trained ResNet50 model for video embeddings
resnet_model = ResNet50(weights='imagenet')
video_model = Model(inputs=resnet_model.input, outputs=resnet_model.layers[-2].output)

# Initialize distributed training strategy
strategy = tf.distribute.MirroredStrategy()

print(f'Number of devices: {strategy.num_replicas_in_sync}')

with strategy.scope():
    def load_and_preprocess_video_embeddings(video_path, resize_dim=(224, 224), skip_frames=5, batch_size=10):
        cap = cv2.VideoCapture(video_path)
        video_embeddings = []

        frame_count = 0
        batch_frames = []

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1
            if frame_count % skip_frames != 0:
                continue

            # Resize the frame to a common size
            frame = cv2.resize(frame, resize_dim)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame / 255.0

            batch_frames.append(frame)

            if len(batch_frames) == batch_size:
                # Process the batch and obtain video embeddings
                batch_frames_array = np.array(batch_frames)
                video_embedding_batch = video_model.predict(batch_frames_array)

                # Extend the list of video embeddings
                video_embeddings.extend(video_embedding_batch)

                # Clear the batch
                batch_frames = []

        # Release the video capture object
        cap.release()

        return video_embeddings

# Add a new column 'comments' to store comments loaded from files
df['video_embedding'] = df['video_name'].apply(load_and_preprocess_video_embeddings)


# Save to HDF5 file
with h5py.File('video_embeddings.h5', 'w') as hf:
    # Save video_description_embedding column
    hf.create_dataset('video_embedding', data=np.vstack(df['video_embedding']))

#testframe = load_and_preprocess_video_embeddings('video_1.mp4')
#print(testframe)

