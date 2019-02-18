import json

import nltk
from nltk.stem.porter import *


documents = []
new_doc = ""
doc_name = ""	
STOPWORDS = []


FILE_PATH = "./time/TIME.ALL"
FILE_BREAKER = "*TEXT"

with open('./time/TIME.STP') as stop_file:
	for line in stop_file.readlines():
		if len(line.strip()) > 0:
			STOPWORDS.append(line.strip())

with open(FILE_PATH, 'r') as file_of_docs:
	for line in file_of_docs.readlines():
		if line.startswith(FILE_BREAKER):
			documents.append((doc_name, new_doc))
			doc_name = line.strip()
			new_doc = ""
			continue
		new_doc += " " + line.strip()
	documents.pop(0)
	documents.append((doc_name, new_doc))

STEMMER = PorterStemmer()

TERMS_IDX = dict()
DOCS_IDX = dict()
doc_idx = 0
for iterator in documents:
	doc_id = iterator[0]
	doc_content = iterator[1]
	DOCS_IDX[doc_idx] = [STEMMER.stem(t) for t in nltk.word_tokenize(doc_content) if t not in STOPWORDS]
	tf_doc = dict()
	for token in [t for t in nltk.word_tokenize(doc_content)]:
		if token in STOPWORDS:
			continue
		token = STEMMER.stem(token)
		if not token in TERMS_IDX:
			TERMS_IDX[token] = dict()
			TERMS_IDX[token]["df"] = 0
		if not doc_idx in TERMS_IDX[token]:
			TERMS_IDX[token]["df"] += 1
			TERMS_IDX[token][doc_idx] = dict()
			TERMS_IDX[token][doc_idx]["tf"] = 0
		TERMS_IDX[token][doc_idx]["tf"] += 1
	doc_idx += 1

with open('terms_info.json', 'w') as fp:
    json.dump(TERMS_IDX, fp)
with open('docs_info.json', 'w') as fp:
    json.dump(DOCS_IDX, fp)
