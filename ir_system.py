import json
from math import sqrt, log2
import nltk
from nltk.stem.porter import *


def get_stopwords(stop_words_path='./time/TIME.STP'):
	with open(stop_words_path) as stop_file:
		return [line.strip().lower() for line in stop_file.readlines() if len(line.strip()) > 0]

stemmer = PorterStemmer()
stop_words = get_stopwords()
def get_list_of_queries(queries_limit=10, queries_path='./time/TIME.QUE', queries_separator='*FIND'):
	documents = []
	new_doc = ""
	doc_name = ""
	with open(queries_path, 'r') as file_of_docs:
		i = 0
		for line in file_of_docs.readlines():
			if line.startswith(queries_separator):
				if new_doc:
					documents.append((doc_name, new_doc))
					i += 1
				doc_name = line.strip()
				new_doc = ""
				if i == queries_limit:
					break
				continue
			new_doc += " " + line.strip().lower()
	return documents

def get_dict_from_json(filename="terms_info.json"):
	with open(filename) as f:
		return json.load(f)

def process_document(document):
	return [stemmer.stem(t) for t in nltk.word_tokenize(document) if t.lower() not in stop_words]

def cosine_sim(index_reader, document, query, doc_id, N):
	dot_prod = 0
	norm_q = 0
	norm_d = 0
	for term in query:
		if term in index_reader and doc_id in index_reader[term]:
			dot_prod += ((index_reader[term][doc_id]["tf"]) * log2(float(N) / float(index_reader[term]["df"])))
		norm_q += 1 if index_reader.get(term) != None else 0
	for term in document:
		norm_d +=  (index_reader[term][doc_id]["tf"] ** 2 )
	return float(dot_prod) / sqrt(norm_d * norm_q)

def search(index_reader, query):
	ans = []
	documents = get_dict_from_json("docs_info.json")
	N = len(documents)
	for doc_id in documents:
		document_content = process_document(documents[doc_id])
		sim = cosine_sim(index_reader, document_content, query, doc_id, N)
		if sim > 0:
			ans.append((sim, int(doc_id) + 1))
	return sorted(ans, reverse=True)



def main():
	queries = get_list_of_queries()
	index_reader = get_dict_from_json()
	qrel ={1:[268,288,304,308,323,326,334],2:[326,334],3:[326,350,364,385],4:[370,378,385,409,421],5:[359,370,385,397,421],6:[257,268,288,304,308,323,324,326,334],7:[386,408],8:[339,358],9:[61,155,156,242,269,315,339,358],10:[61,156,242,269,339,358],11:[195,198],12:[61,155,156,242,269,339,358],13:[87,170,185],14:[269],15:[94,118,128,164,424],16:[169,170,239],17:[303,358],18:[356],19:[99,100,195,267,344],20:[356],21:[305,318],22:[41,356],23:[425],24:[318],25:[342],26:[318,425],27:[272,295,306],28:[189,219,265,277,360],29:[192,329],30:[253,261,271,293,325],31:[47,56,81,103,150,183,291],32:[279],33:[251,335],34:[294],35:[318],36:[217],37:[403,406],38:[364],39:[22,73,173,189,219,265,277,360,396],40:[20,71,131,148,182,207,261,272,325],41:[23,47,53,54,174,315],42:[48],43:[329,341],44:[295,316],45:[58,71,148,365,381],46:[1,20,23,32,39,47,53,54,80,93,151,157,174,202,272,291,294,348],47:[23,47,48,53,54,56],48:[306],49:[47,56,81,103,150,183,205,291],50:[157],51:[186,190,276],52:[186,259],53:[295,306],54:[315,403],55:[1,47,54,89,135,157,228,247,254,272,360,404],56:[420],57:[79,306],58:[1,47,54,89,135,157,247,254],59:[38,79],60:[79,315],61:[1,23,38,47,135,157,174,193,228,247,254,295,315,343,363],62:[233],63:[35,86,100,121,159,194,210,224,309,379,388],64:[322,328],65:[342],66:[100,115],67:[7,63,184],68:[7,8,31,46,92,307,328,422],69:[70,100,115,121,139,159,194,210,224,234,309,379,388],70:[226],71:[117,410,419],72:[338],73:[320],74:[25,285],75:[390],76:[168,227,230,236,338],77:[318],78:[258,410],79:[320],80:[27,68,78,170,185,275,279,280,290,306,307,315,341,343,401,413,419],81:[199,380],82:[280,307,315,343,413],83:[343,363]}
	index_q = 1
	for q in queries:
		query_content = q[1]
		query_content = process_document(query_content)
		results = search(index_reader, query_content)
		print(q)
		counter = 1
		for i in results:
			flag = ""
			if i[1] in qrel[index_q]:
				flag = "*"
			print("{}.- {} {} {}".format(counter, i[1], i[0], flag))
			counter += 1
		index_q += 1


if __name__ == '__main__':
	main()