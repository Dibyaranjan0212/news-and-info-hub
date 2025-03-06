import requests
from lxml import html
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import numpy as np
import re
import nltk
import time
import textstat
from transformers import BartForConditionalGeneration, BartTokenizer

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

def search_bing(query, num_results=5, retries=3):
    search_url = "https://www.bing.com/search"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    params = {'q': query, 'count': num_results}
    
    for attempt in range(retries):
        try:
            response = requests.get(search_url, headers=headers, params=params)
            response.raise_for_status()
            
            tree = html.fromstring(response.content)
            urls = []
            for result in tree.xpath('//li[@class="b_algo"]/h2/a/@href'):
                if not any(keyword in result for keyword in [
                    "youtube.com", "twitter.com", 
                    "facebook.com", "instagram.com", "linkedin.com", 
                    "pinterest.com", "tiktok.com"
                ]):
                    urls.append(result)
                    if len(urls) >= num_results:
                        break
            
            return urls
        
        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error: {e}")
            if e.response.status_code == 429:
                print("Rate limit hit, retrying...")
                time.sleep(5)
            else:
                raise
        except Exception as e:
            print(f"An error occurred: {e}")
            raise

def extract_and_filter_text(url, headers):
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        tree = html.fromstring(response.content)
        paragraphs = tree.xpath('//p/text()')
        
        filtered_texts = []
        for paragraph in paragraphs:
            paragraph_cleaned = re.sub(r'\s+', ' ', paragraph).strip()
            filtered_texts.append(paragraph_cleaned)
        
        return ' '.join(filtered_texts)
    
    except Exception as e:
        print(f"Failed to extract text from {url}: {e}")
        return ""

def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if word.lower() not in stop_words and word.isalnum()]
    return ' '.join(tokens)

def apply_topic_modeling(documents, n_topics=1, model_type="LDA"):
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))  # Using n-grams (1 and 2 words)
    X_tfidf = vectorizer.fit_transform(documents)
    
    terms = vectorizer.get_feature_names_out()
    
    if model_type == "LDA":
        model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        model.fit(X_tfidf)
        components = model.components_
    
    elif model_type == "NMF":
        model = NMF(n_components=n_topics, random_state=42)
        model.fit(X_tfidf)
        components = model.components_
    
    elif model_type == "LSA":
        model = TruncatedSVD(n_components=n_topics, random_state=42)
        model.fit(X_tfidf)
        components = model.components_
    
    stop_words = set(stopwords.words('english'))  # Remove stopwords from topic words
    topics = []
    
    for idx, topic in enumerate(components):
        topic_words = [terms[i] for i in topic.argsort()[:-10 - 1:-1] if terms[i] not in stop_words]
        topics.append(topic_words)
        print(f"Topic {idx + 1} ({model_type}):", topic_words)
    
    return topics

def abstractive_summarization_bart(text, max_length=150, min_length=30):
    bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    inputs = bart_tokenizer([text], max_length=1024, return_tensors='pt', truncation=True)
    summary_ids = bart_model.generate(inputs['input_ids'], max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def rank_websites_by_readability(texts):
    ranked_urls = sorted(texts, key=lambda x: textstat.flesch_reading_ease(x[1]), reverse=True)
    return ranked_urls

    ranked_websites = rank_websites_by_readability(website_texts)
    
    print("\nRanked websites based on readability score:")
    for idx, (url, _) in enumerate(ranked_websites):
        print(f"{idx + 1}. {url}")

    preprocessed_text = preprocess_text(combined_text)
    sentences = re.split(r'(?<=[.!?]) +', combined_text)
    
    print("\nApplying LDA...")
    lda_topics = apply_topic_modeling([preprocessed_text], n_topics=1, model_type="LDA")

    print("\nApplying NMF...")
    nmf_topics = apply_topic_modeling([preprocessed_text], n_topics=1, model_type="NMF")

    print("\nApplying LSA...")
    lsa_topics = apply_topic_modeling([preprocessed_text], n_topics=1, model_type="LSA")
    
    bart_summary = abstractive_summarization_bart(combined_text)
    
    print("\nAbstractive Summary (BART):")
    print(bart_summary)


