#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask
from flask import jsonify
from flask import Blueprint, request, jsonify, make_response
import requests
import re
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from transformers import BertTokenizer, TFBertModel

app = Flask(__name__)
def scrape_youtube_links(link):
    response = requests.get(link)
    html_content = response.text

    # Use regular expressions to find and extract all video IDs
    video_ids = re.findall(r'{"videoRenderer":{.*?"videoId":"([a-zA-Z0-9_-]+)"', html_content)

    if video_ids:
        return video_ids
    else:
        print("No video IDs found.")
        return None
def scrape_youtube_data(video_id):
    youtube_url = f"https://www.youtube.com/watch?v={video_id}"
    response = requests.get(youtube_url)
    html_content = response.text
    #print(html_content)
    
    # Use regular expressions to extract the comments count
    comments_match = re.search(r'"commentCount":{"simpleText":"(\d+(\.\d+)?[KkMm]?)', html_content)
    
    subscriber_count_match = re.search(r'"simpleText":"([\d.]+[KkMm]?) subscribers"', html_content)

    if comments_match:
        comments_text = comments_match.group(1)
        #print(comments_text)
    # Convert the text representation of comments count to an integer
        if 'K' in comments_text.upper():
            comments_count = int(float(comments_text[:-1]) * 1000)
        elif 'M' in comments_text.upper():
            comments_count = int(float(comments_text[:-1]) * 1000000)
        else:
            comments_count = int(comments_text)
    else:
        comments_count = 0
            
    if subscriber_count_match:
        subscriber_text = subscriber_count_match.group(1)
        if 'K' in subscriber_text.upper():
            subscriber_count = int(float(subscriber_text[:-1]) * 1000)
        elif 'M' in subscriber_text.upper():
            subscriber_count = int(float(subscriber_text[:-1]) * 1000000)
        else:
            subscriber_count = int(comments_text)
    else:
        subscriber_count = 0
        
    likes_match = re.search(r'"label":"([\d,]+) likes"', html_content)

    if likes_match:
        num_likes = likes_match.group(1)
    # Remove commas from the extracted number
        num_likes = num_likes.replace(',', '')
    else:
        num_likes = 0
        
    views_match = re.search(r'"simpleText":"([\d,]+) views"', html_content)

    if views_match:
        view_count = views_match.group(1)
        # Remove commas from the extracted number
        view_count = view_count.replace(',', '')
    else:
        view_count = 0
        
    
    title_match = re.search(r'"title":{"runs":\[{"text":"(.*?)"', html_content)

    
    if title_match:
        title = title_match.group(1)
    else:
        title = ""
    
    description_match = re.search(r'"description":{"simpleText":"(.*?)"', html_content)
    if description_match:
        description = description_match.group(1)
    else:
        description = ""
    
    channel_name_match = re.search(r'"channelName":"(.*?)"', html_content)
    if channel_name_match:
        channel_name = channel_name_match.group(1)
    else:
        channel_name = ""
    
    comments = ""
    
    return {
        'video_id': video_id,
        'view_count': view_count,
        'num_likes': num_likes,
        'num_subscribers': subscriber_count,
        'num_comments': comments_count,
        'title': title,
        'description': description,
        'channel_name': channel_name,
        'comments': comments
    }
def scrape_multiple_youtube_data(video_ids):
    data_list = []
    for video_id in video_ids:
        data = scrape_youtube_data(video_id)
        data_list.append(data)

    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(data_list)
    return df
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
def get_predictions(row):
    video_description = row['description']
    channel_name = row['channel_name']
    video_title = row['title']
    comments = row['comments']
    num_subsribers = row['num_subscribers']
    num_comments = row['num_comments']
    view_count = row['view_count']
    num_likes = row['num_likes']
    
    #print(channel_name , comments)

    desc_embedding = text_embedding(video_description)
    channel_embedding = text_embedding(channel_name)
    title_embedding = text_embedding(video_title)
    comments_embedding = text_embedding(comments)
    

    X_text_combined = np.concatenate((desc_embedding, channel_embedding, comments_embedding, title_embedding), axis=1)
    X_combined = np.concatenate((X_text_combined, np.array([[num_subsribers, num_comments, view_count, num_likes]])), axis=1)
    # Load the model
    deception_model = tf.keras.models.load_model('deception_model_dense')
    predictions = deception_model.predict(X_combined)

    threshold = 0.5
    predicted_class = "Non-deceptive" if predictions[0, 0] >= threshold else "Deceptive"
    
    return row['video_id'],predicted_class

@app.route('/api/deception', methods=['GET'])
def decp():
    youtube_link = request.args.get('url')
    #youtube_link = "https://www.youtube.com/results?search_query=biggboss"
    results = scrape_youtube_links(youtube_link)
    result_df = scrape_multiple_youtube_data(results)
    #print(result_df)
    # Load the model
    result_df['num_likes'] = pd.to_numeric(result_df['num_likes'], errors='coerce', downcast='integer')
    result_df['view_count'] = pd.to_numeric(result_df['view_count'], errors='coerce', downcast='integer')
    deception_model = tf.keras.models.load_model('deception_model')
    # Assuming `results_df` is your DataFrame containing YouTube data
    predictions_list = result_df.apply(get_predictions, axis=1)
    result = predictions_list.tolist()
    return jsonify(result)

if __name__ == '__main__':
    app.run()


# In[ ]:




