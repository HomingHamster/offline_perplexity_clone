import requests
from bs4 import BeautifulSoup
import numpy as np  # Required to dedupe sites
from langchain.embeddings import LlamaCppEmbeddings
from langchain.llms.llamacpp import LlamaCpp
from readabilipy import simple_json_from_html_string  # Required to parse HTML to pure text
from langchain.schema import Document  # Required to create a Document object
from langchain.vectorstores import Chroma
from langchain.chains import VectorDBQA  # Required to create a Question-Answer object using a vector
import pprint  # Required to pretty print the results
from langchain.text_splitter import CharacterTextSplitter

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"
}

params = {
    "q": 'history of the human genome project', #this is the first query which includes the precise subject that your followup questions are related to
    "hl": "en",
    "gl": "uk",
    "start": 0,
}

page_limit = 10
page_num = 0
urls=[]

while True:
    page_num += 1
    print(f"page: {page_num}")
    html = requests.get("https://www.google.com/search", params=params, headers=headers, timeout=30)
    soup = BeautifulSoup(html.text, 'lxml')
    for result in soup.select(".tF2Cxc"):
        link = result.select_one(".yuRUbf a")["href"]
        urls.append(link.split("#")[0])
    if page_num == page_limit:
        break
    if soup.select_one(".d6cvqb a[id=pnnext]"):
        params["start"] += 10
    else:
        break

urls = list(np.unique(urls))
print(urls)

#if the scrape of google returns results that cant be parsed into json, then copy the modified list here (better parsing is coming)

#urls=['https://embryo.asu.edu/pages/human-genome-project-1990-2003', 
      #'https://en.wikipedia.org/wiki/Human_Genome_Project', 
      #'https://plato.stanford.edu/entries/human-genome/']

def scrape_and_parse(url):
    """Scrape a webpage and parse it into a Document object"""
    req = requests.get(url)
    article = simple_json_from_html_string(req.text, use_readability=True)
    return Document(page_content=article['plain_text'][0]['text'], metadata={'source': url, 'page_title': article['title']})

documents = [scrape_and_parse(f) for f in urls]  # Scrape and parse all the urls
print(documents)

text_splitter = CharacterTextSplitter(separator=' ', chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)
len(texts)

#download the mistril or another model and modify the path here
embeddings = LlamaCppEmbeddings(model_path="/home/ubuntu/perp/mistral-7b-v0.1.Q5_K_M.gguf", n_ctx=2048)
llm = LlamaCpp(
    model_path="/home/ubuntu/perp/mistral-7b-v0.1.Q5_K_M.gguf", #modify the path here
    temperature=0.75,
    n_ctx=2048,
)

docsearch = Chroma.from_documents(texts, embeddings)

# First followup question
qa = VectorDBQA.from_chain_type(llm=llm, chain_type="stuff", vectorstore=docsearch, return_source_documents=True)
result = qa({"query": "Who were the main players in the race to complete the human genome? And what were their"
                      " approaches? Give as much detail as possible."})
pprint.pprint(result)

#second followup question
qa = VectorDBQA.from_chain_type(llm=llm, chain_type="stuff", vectorstore=docsearch, return_source_documents=True)
result = qa({"query": "How were the donor participants recruited for the human genome project? "
                      "Summarize in three sentences."})

pprint.pprint(result)