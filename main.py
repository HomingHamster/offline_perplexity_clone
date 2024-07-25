import requests
from bs4 import BeautifulSoup
import numpy as np  # Required to dedupe sites
from readabilipy import simple_json_from_html_string  # Required to parse HTML to pure text
from langchain.schema import Document  # Required to create a Document object
from langchain.chains import VectorDBQA  # Required to create a Question-Answer object using a vector
import pprint  # Required to pretty print the results
from langchain.text_splitter import CharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"
}

params = {
    "q": 'history of the human genome project', #this is the first query which includes the precise subject that your followup questions are related to
    "hl": "en",
    "gl": "uk",
    "start": 0,
}

page_limit = 1
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


documents = []
for url in urls:
    req = requests.get(url)
    article = simple_json_from_html_string(req.text, use_readability=True)
    if article["plain_text"]:
        documents += [Document(page_content=article['plain_text'][0]['text'],
                        metadata={'source': url, 'page_title': article['title']})]

print(documents)

text_splitter = CharacterTextSplitter(separator=' ', chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)
len(texts)

#download mistral with ollama pull mistral
embeddings = OllamaEmbeddings(model="tinyllama")
llm = ChatOllama(
    model="tinyllama",
    temperature=0,
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