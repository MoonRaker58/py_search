import requests
from bs4 import BeautifulSoup
import spacy

# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")

def get_website_text(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    # Extract text content from HTML
    text = ' '.join([p.text for p in soup.find_all('p')])
    return text

def perform_semantic_search(website_url, topics):
    # Get the entire text content of the website
    website_text = get_website_text(website_url)

    # Process the website text using spaCy
    website_doc = nlp(website_text)

    # Process the topics using spaCy
    topic_docs = [nlp(topic) for topic in topics]

    # Perform semantic search
    results = []
    for topic_doc in topic_docs:
        for sent in website_doc.sents:
            similarity = topic_doc.similarity(sent)
            if similarity > 0.5:  # You can adjust the similarity threshold
                results.append({
                    'topic': topic_doc.text,
                    'sentence': sent.text,
                    'similarity': similarity
                })

    return results

if __name__ == "__main__":
    # Specify the website URL and topics of interest
    website_url = "https://www.freight360.net/resources/blog/"
    topics_of_interest = ["Freight Broker Business Models", "Ownership of a Freight Brokerage", "Line of Credit"]

    # Perform semantic search
    search_results = perform_semantic_search(website_url, topics_of_interest)

    # Display the results
    for result in search_results:
        print(f"Topic: {result['topic']}")
        print(f"Sentence: {result['sentence']}")
        print(f"Similarity: {result['similarity']:.2f}\n")
