import time
import json
from math import sqrt, log10
import nltk
from nltk.stem.porter import *


def get_stopwords(stop_words_path='./time/TIME.STP'):
	with open(stop_words_path) as stop_file:
		return [line.strip().lower() for line in stop_file.readlines() if len(line.strip()) > 0]

stemmer = PorterStemmer()
stop_words = get_stopwords()
qrel ={1:[268,288,304,308,323,326,334],2:[326,334],3:[326,350,364,385],4:[370,378,385,409,421],5:[359,370,385,397,421],6:[257,268,288,304,308,323,324,326,334],7:[386,408],8:[339,358],9:[61,155,156,242,269,315,339,358],10:[61,156,242,269,339,358],11:[195,198],12:[61,155,156,242,269,339,358],13:[87,170,185],14:[269],15:[94,118,128,164,424],16:[169,170,239],17:[303,358],18:[356],19:[99,100,195,267,344],20:[356],21:[305,318],22:[41,356],23:[425],24:[318],25:[342],26:[318,425],27:[272,295,306],28:[189,219,265,277,360],29:[192,329],30:[253,261,271,293,325],31:[47,56,81,103,150,183,291],32:[279],33:[251,335],34:[294],35:[318],36:[217],37:[403,406],38:[364],39:[22,73,173,189,219,265,277,360,396],40:[20,71,131,148,182,207,261,272,325],41:[23,47,53,54,174,315],42:[48],43:[329,341],44:[295,316],45:[58,71,148,365,381],46:[1,20,23,32,39,47,53,54,80,93,151,157,174,202,272,291,294,348],47:[23,47,48,53,54,56],48:[306],49:[47,56,81,103,150,183,205,291],50:[157],51:[186,190,276],52:[186,259],53:[295,306],54:[315,403],55:[1,47,54,89,135,157,228,247,254,272,360,404],56:[420],57:[79,306],58:[1,47,54,89,135,157,247,254],59:[38,79],60:[79,315],61:[1,23,38,47,135,157,174,193,228,247,254,295,315,343,363],62:[233],63:[35,86,100,121,159,194,210,224,309,379,388],64:[322,328],65:[342],66:[100,115],67:[7,63,184],68:[7,8,31,46,92,307,328,422],69:[70,100,115,121,139,159,194,210,224,234,309,379,388],70:[226],71:[117,410,419],72:[338],73:[320],74:[25,285],75:[390],76:[168,227,230,236,338],77:[318],78:[258,410],79:[320],80:[27,68,78,170,185,275,279,280,290,306,307,315,341,343,401,413,419],81:[199,380],82:[280,307,315,343,413],83:[343,363]}
global flag
flag = True
global test

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
			idf = log10(N / index_reader[term]["df"])
			tf = index_reader[term][doc_id]["tf"]
			dot_prod += (query[term] * tf * idf)
		norm_q += query[term] ** 2 if index_reader.get(term) != None else 0
	for term in set(document):
		idf = log10(N / index_reader[term]["df"])
		tf = index_reader[term][doc_id]["tf"]
		w_id = tf * idf
		norm_d += (w_id ** 2)
	return dot_prod / sqrt(norm_d * norm_q)

def search(index_reader, query, limit=100):
	ans = []
	documents = get_dict_from_json("docs_info.json")
	N = len(documents)
	t1 = time.time()
	for doc_id in documents:
		t = time.time()
		document_content = documents[doc_id]
		t = time.time()
		sim = cosine_sim(index_reader, document_content, query, doc_id, N)
		if sim > 0:
			ans.append((sim, int(doc_id) + 1))
	print("Tiempo en realizar la consulta {} ". format(time.time() - t1))
	ans = sorted(ans, reverse=True)
	ans = ans[:100] if len(ans) > 100 else ans
	return ans

def evaluator(results, query_idx):
	precision = recall = retrieved_docs = 0
	relevant_documents = len(qrel.get(query_idx))
	retrieved_and_relevant_docs = 0
	precision_list = []
	relevant_boolean_list = []
	for r in results:
		doc_id = r[1]
		score = r[0]
		retrieved_docs += 1
		if doc_id in qrel[query_idx]:
			retrieved_and_relevant_docs += 1
			relevant_boolean_list.append(1)
		else:
			relevant_boolean_list.append(0)
		precision = retrieved_and_relevant_docs / retrieved_docs
		precision_list.append(precision)
		recall = retrieved_and_relevant_docs / relevant_documents
		ap = 0
		for p_i, rel_i in zip(precision_list, relevant_boolean_list):
			ap += p_i * rel_i
		ap /= relevant_documents
		results[retrieved_docs - 1] += (recall, precision, ap) 
	return results

def save_html(results, queries, html_filename='index.html', table_color='#4CAF50'):
	begin_html = '<!DOCTYPE html><html><head><meta name="viewport" content="width=device-width, initial-scale=1"><style>table{{border-collapse: collapse; font-family: "Trebuchet MS", Arial, Helvetica, sans-serif; width: 50%; margin: 0px auto;}}h2{{font-family: "Trebuchet MS", Arial, Helvetica, sans-serif;}}th, td{{text-align: left; padding: 8px; border: 1px solid #ddd;}}tr:nth-child(even){{background-color: #f2f2f2;}}tr:hover{{background-color: #ddd;}}th{{padding-top: 12px; padding-bottom: 12px; text-align: left; background-color: {}; color: white;}}</style></head><body>'.format(table_color)
	end_html = '</body></html>'
	with open(html_filename, 'w') as f:
		f.write(begin_html)
		query_res_idx = 0
		for q in queries:
			f.write('<h2>{}:{}</h2>'.format(q[0], q[1]))
			f.write('<div style="overflow-x:auto;"> <table> <tr> <th>ID</th> <th>Document</th> <th>Cosine Sim</th> <th>Recall</th> <th>Precision</th> <th>AP</th> </tr>')
			counter = 1
			for i in results[query_res_idx]:
				flag = 'style="font-weight:bold"' if i[1] in qrel[query_res_idx + 1] else ""
				f.write('<tr {}><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>'.format(flag, counter, i[1], i[0], i[2], i[3], i[4]))
				counter += 1
			f.write('</table></div>')
			query_res_idx += 1
		f.write(end_html)
def create_sum_of_doc_vector_with_dicts(index_reader, documents, docs_ids):
	doc_sum = {}
	N = len(documents)
	for doc_id in docs_ids:
		document_content = documents[doc_id]
		for term in document_content:
			idf = log10(N / index_reader[term]["df"])
			tf = index_reader[term][doc_id]["tf"]
			if term in doc_sum:
				doc_sum[term] += (idf * tf)
			else:
				doc_sum[term] = idf * tf
	return doc_sum
def scalar_times_dict(scalar, vector_as_dict):
	for key in vector_as_dict:
		vector_as_dict[key] *= scalar

def vector_dict_add(dict_a, dict_b):
	set_of_terms = set(dict_a).union(set(dict_b))
	ans = {}
	for term in set_of_terms:
		a_i = dict_a.get(term, 0)
		b_i = dict_b.get(term, 0)
		ans[term] = a_i + b_i
	return ans
def build_query_using_rocchio(index_reader, query, number_of_relevant_docs, alpha=1, beta=0.8, gamma=0.1):
	documents = get_dict_from_json("docs_info.json")
	results = search(index_reader, query)
	number_of_relevant_docs = number_of_relevant_docs if number_of_relevant_docs < len(results) else len(results)
	relevant_documents = [str(results[i][1] - 1) for i in range(number_of_relevant_docs)]
	non_relevant_docs = [str(results[i][1] - 1) for i in range(number_of_relevant_docs, len(results))]
	vector_original_query = { term : alpha for term in set(query)}
	sum_of_relevant_docs = create_sum_of_doc_vector_with_dicts(index_reader, documents, relevant_documents)
	sum_of_non_relevant_docs = create_sum_of_doc_vector_with_dicts(index_reader, documents, non_relevant_docs)
	scalar_times_dict(beta / number_of_relevant_docs, sum_of_relevant_docs)
	number_of_non_relevant_docs = len(results) - number_of_relevant_docs
	scalar_times_dict(- gamma / number_of_non_relevant_docs, sum_of_non_relevant_docs)
	modified_query = vector_dict_add(vector_original_query, sum_of_relevant_docs)
	modified_query = vector_dict_add(modified_query, sum_of_non_relevant_docs)
	return modified_query

def solution_task2():
	queries = get_list_of_queries()
	index_reader = get_dict_from_json()
	all_results = []
	index_q = 1
	for q in queries:
		query_content = q[1]
		query_content = process_document(query_content)
		query_content = { term : 1 for term in set(query_content)}
		results = search(index_reader, query_content)
		results = evaluator(results, index_q)
		index_q += 1
		all_results.append(results)
	save_html(all_results, queries, "no_query_expansion.html")

def solution_task3():
	queries = get_list_of_queries()
	index_reader = get_dict_from_json()
	all_results = []
	index_q = 1
	for q in queries:
		query_content = process_document(q[1])
		query_content = { term : 1 for term in set(query_content)}
		modified_query = build_query_using_rocchio(index_reader, query_content, 3)
		results = search(index_reader, modified_query)
		results = evaluator(results, index_q)
		index_q += 1
		all_results.append(results)
	save_html(all_results, queries, "query_expansion_using_rocchio.html", '#69aa96')

if __name__ == '__main__':
	solution_task2()
	solution_task3()