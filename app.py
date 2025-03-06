from flask import Flask, render_template, request
from search import search_bing, extract_and_filter_text, preprocess_text, apply_topic_modeling, abstractive_summarization_bart, rank_websites_by_readability

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    query = ""
    results = []
    lda_topics = []
    nmf_topics = []
    lsa_topics = []
    summary = ""
    
    if request.method == 'POST':
        query = request.form['keyword']
        
        search_results = search_bing(query)
        
        if search_results:
            combined_text = ''
            website_texts = []
            for url in search_results:
                filtered_text = extract_and_filter_text(url, {'User-Agent': 'Mozilla/5.0'})
                website_texts.append((url, filtered_text))
                combined_text += filtered_text + " "
                
            ranked_websites = rank_websites_by_readability(website_texts)

            preprocessed_text = preprocess_text(combined_text)

            lda_topics = apply_topic_modeling([preprocessed_text], n_topics=3, model_type="LDA")
            nmf_topics = apply_topic_modeling([preprocessed_text], n_topics=3, model_type="NMF")
            lsa_topics = apply_topic_modeling([preprocessed_text], n_topics=3, model_type="LSA")

            summary = abstractive_summarization_bart(combined_text)

            results = [url for url, _ in ranked_websites]

    return render_template('index.html', query=query, results=results, lda_topics=lda_topics, nmf_topics=nmf_topics, lsa_topics=lsa_topics, summary=summary)

if __name__ == '__main__':
    app.run(debug=True)
