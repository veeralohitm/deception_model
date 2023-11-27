from flask import Flask
from flask import jsonify
from flask import Blueprint, request, jsonify, make_response
# Imports
import requests
from bs4 import BeautifulSoup
import transformers
from transformers import pipeline

app = Flask(__name__)
@app.route('/hello/', methods=['GET', 'POST'])
def welcome():
    return "Hello World!"


@app.route('/person/')
def hello():
    return jsonify({'name':'Jimit',
                    'address':'India'})


# create a route for our API
@app.route('/api/summarize', methods=['GET'])
def data():
  # get query parameters
  url = request.args.get('url')
  page = requests.get(url).text
    # Turn page into BeautifulSoup object to access HTML tags
  soup = BeautifulSoup(page)
    # Get headline
  headline = soup.find('h1').get_text()
    # Get text from all <p> tags
  p_tags = soup.find_all('p')
    # Get the text from each of the “p” tags and strip surrounding whitespace.
  p_tags_text = [tag.get_text().strip() for tag in p_tags]
  print('for loop tags')
    # Filter out sentences that contain newline characters '\n' or don't contain periods.
  sentence_list = [sentence for sentence in p_tags_text if not '\n' in sentence]
  sentence_list = [sentence for sentence in sentence_list if '.' in sentence]
    # Combine list items into string.
  article = ' '.join(sentence_list)
    #print(article)
  ar_len=len(article.split())
  print ("The number of words in the article is : " +  str(ar_len))
  summarizer = pipeline("summarization", model="stevhliu/my_awesome_billsum_model")
  print("pipeline")
  #article = "The webpage consist of 4 checkboxes , 3 buttons , 2 searches and 4 links"
  summarized = summarizer(article, min_length=75, max_length=300)
  summarized = summarizer(article)
  print("summarized")
  print(summarized)
  summ=' '.join([str(i) for i in summarized])
  summ=summ.replace("{","")
  summ=summ.replace("''","")
  summ=summ.replace("\x92","")
  #print(summ)
  summ1=len(summ.split())
  print(" The no. of words in the summarization using huggingface transformers is :" +str(summ1))
  print(summarized)
  return jsonify(summarized)


if __name__ == '__main__':
    app.run()