from flask import Flask, render_template, request, url_for, redirect
from wtforms import Form, TextAreaField, validators
from bs4 import BeautifulSoup
from nltk.stem.porter import PorterStemmer
import pickle
import os
import numpy as np
import re

from database import Database

cur_dir = os.path.dirname(__file__)

invite = pickle.load(open(os.path.join('pkl_objects', 'invite.pkl'), 'rb'))
like= pickle.load(open(os.path.join('pkl_objects', 'like.pkl'), 'rb'))
complain = pickle.load(open(os.path.join('pkl_objects', 'complain.pkl'), 'rb'))
query= pickle.load(open(os.path.join('pkl_objects', 'query.pkl'), 'rb'))
model = pickle.load(open(os.path.join('pkl_objects', 'model.pkl'), 'rb'))
model_l = pickle.load(open(os.path.join('pkl_objects', 'model_l.pkl'), 'rb'))
list_stop = pickle.load(open(os.path.join('pkl_objects', 'stopwords.pkl'), 'rb'))


db = os.path.join(cur_dir, 'reviews.sqlite')

num_features = 300


def generate_words(text, remove_stopwords=False):
	review = BeautifulSoup(text).get_text()
	review = re.sub("[^a-zA-Z]", " ", review)
	words = review.lower().split()
	if remove_stopwords:
		stops = list_stop
		words = [z for z in words if z not in stops]

	return words


def featureVecMethod(words, model, num_features):
	featureVec = np.zeros(num_features, dtype="float64")
	nwords = 0

	index2word_set = set(model.wv.index2word)

	for word in words:
		if word in index2word_set:
			nwords = nwords + 1
			featureVec = np.add(featureVec, model[word])

	featureVec = np.divide(featureVec, nwords)
	return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
	counter = 0
	reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float64")
	for review in reviews:
		reviewFeatureVecs[counter] = featureVecMethod(review, model, num_features)
		counter = counter + 1

	return reviewFeatureVecs


def classify(document):
	label = {0: 'False', 1: 'True'}
	#testcorp = []
	#testcorp.append(generate_words(str(document), remove_stopwords=True))
	#document = getAvgFeatureVecs(testcorp, model, 300)
	#document = Imputer().fit_transform(document)
	corpus_x = []
	review = re.sub('[^a-zA-Z]', ' ', str(document))
	review = review.lower()
	review = review.split()
	ps = PorterStemmer()
	review = [ps.stem(word) for word in review if not word in list_stop]
	review = ' '.join(review)
	corpus_x.append(review)
	doc=model.transform(corpus_x).toarray()
	doc_l=model_l.transform(corpus_x).toarray()
	p1 = invite.predict(doc)[0]
	p2 = like.predict(doc_l)[0]
	p4 = query.predict(doc)[0]
	p3 = not p2
	p1 = label[p1]
	p2 = label[p2]
	p3 = label[p3]
	p4 = label[p4]
	return p1, p2, p3, p4


def train(document, y1, y2, y3, y4):
	#testcorp = []
	#testcorp.append(generate_words(str(document), remove_stopwords=True))
	#document = getAvgFeatureVecs(testcorp, model, 300)
	#document = Imputer().fit_transform(document)
	corpus_x = []
	review = re.sub('[^a-zA-Z]', ' ', str(document))
	review = review.lower()
	review = review.split()
	ps = PorterStemmer()
	review = [ps.stem(word) for word in review if word not in list_stop]
	review = ' '.join(review)
	corpus_x.append(review)
	document = model.transform(corpus_x).toarray()
	document_l = model_l.transform(corpus_x).toarray()
	invite.partial_fit(document, [y1], classes=np.unique(y1))
	like.partial_fit(document_l, [y2], classes=None)
	complain.partial_fit(document, [y3], classes=None)
	query.partial_fit(document, [y4], classes=None)


# def sqlite_entry(path, document, y1, y2, y3, y4):
# 	conn = sqlite3.connect(path)
# 	c = conn.cursor()
# 	c.execute("INSERT INTO review_db (review, sent_1, sent_2, sent_3, sent_4)" " VALUES (?, ?, ?, ?, ?)", (document, y1, y2, y3, y4))
# 	conn.commit()
# 	conn.close()

def json(path, document, y1, y2, y3, y4):
	return{
		"path": path,
		"document": document,
		"y1": y1,
		"y2": y2,
		"y3": y3,
		"y4": y4
	}


def save_to_db(path, document, y1, y2, y3, y4):
	Database.insert("user_data", json(path, document, y1, y2, y3, y4))


app = Flask(__name__)


class ReviewForm(Form):
	commentreview = TextAreaField('', [validators.DataRequired()])


@app.before_first_request
def database_initialize():
	Database.initialize()


@app.route('/')
def index():
	form = ReviewForm(request.form)
	return render_template('reviewform.html', form=form)


@app.route('/results', methods=['POST'])
def results():
	form = ReviewForm(request.form)
	if request.method == 'POST' and form.validate():
		review = request.form['commentreview']
		p1, p2, p3, p4 = classify(review)
		return render_template('results.html', content=review, pred1=p1, pred2=p2, pred3=p3, pred4=p4)
	return render_template('reviewform.html', form=form)


@app.route('/feedback', methods=['POST'])
def feedback():
# 	review = request.form['review']
# 	pred1 = request.form['pred1']
# 	pred2 = request.form['pred2']
# 	pred3 = request.form['pred3']
# 	pred4 = request.form['pred4']
# 	label = {'False': 0, 'True': 1}
# 	p1 = label[pred1]
# 	p2 = label[pred2]
# 	p3 = label[pred3]
# 	p4 = label[pred4]
# 	if request.form.get("invite") == "yes":
# 		p1 = 1
# 	else:
# 		p1 = 0
# 	if request.form.get("like") == "yes":
# 		p2 = 1
# 	else:
# 		p2 = 0
# 	if request.form.get("complain") == "yes":
# 		p3 = 1
# 	else:
# 		p3 = 0
# 	if request.form.get("query") == "yes":
# 		p4 = 1
# 	else:
# 		p4 = 0
	#train(review, p1, p2, p3, p4)
# 	save_to_db(db, review, p1, p2, p3, p4)
	form = ReviewForm(request.form)
	return render_template('reviewform.html', form=form)
# 	return render_template('reviewform.html')


@app.route('/about', methods=['GET', 'POST'])
def about():
	if request.method == 'POST':
		return redirect(url_for('index'))
	return render_template('about.html')


if __name__ == '__main__':
	app.run(debug=True)
