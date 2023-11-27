import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
import numpy as np
import cv2
import re

# Import deceptive dataset
df_d = pd.read_csv("deceptive_vid.csv")
# Adding deceptive flag 
df_d['deceptive_flag'] = 0 
#Import non-deceptive datset
df_nd = pd.read_csv("nondeceptive_vid.csv")
# Adding non deceptive flag 
df_nd['deceptive_flag'] = 1

df = pd.concat([df_d, df_nd],ignore_index=True)
# Drop Category
df.drop(columns=['Category'])
# Remove if nan Video title
df = df[df['video_name'].notna()]

df['video_title'] = df['video_title'].astype(str)
df['channel_name'] = df['channel_name'].astype(str)
df['video_description'] = df['video_description'].astype(str)
df['comments'] = df['comments'].astype(str)

def load_comments_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        comments = file.readlines()

    # Extract comments from lines with a regex pattern
    pattern = re.compile(r'\d+: (.+)')
    comments = [pattern.match(comment).group(1) for comment in comments if pattern.match(comment)]

    return comments

def text_embedding(text):
    # BERT Tokenization
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokens = tokenizer(text, return_tensors='tf', max_length=512, truncation=True)

    # BERT Model
    model = TFBertModel.from_pretrained('bert-base-uncased')
    embeddings = model(tokens)['last_hidden_state']
    
    # Consider using the mean or pooling operation over the embeddings
    mean_embedding = tf.reduce_mean(embeddings, axis=1)

    return mean_embedding.numpy()

# Assume you have a pre-trained ResNet50 model for video embeddings
resnet_model = ResNet50(weights='imagenet')
video_model = Model(inputs=resnet_model.input, outputs=resnet_model.layers[-2].output)

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
df['comb_comments'] = df['comments'].apply(lambda x: load_comments_from_file(x))

# Add a new column 'comment_embeddings' to store comment text embeddings
df['comment_embeddings'] = df['comb_comments'].apply(lambda comments: [text_embedding(comment) for comment in comments])

df['title_embedding'] = df['video_title'].apply(text_embedding)
df['channel_name_embedding'] = df['channel_name'].apply(text_embedding)
df['video_description_embedding'] = df['video_description'].apply(text_embedding)
df.to_csv("text_embeddings.csv", index=False)
