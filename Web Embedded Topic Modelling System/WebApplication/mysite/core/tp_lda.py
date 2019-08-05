
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
import pyLDAvis.gensim
import pyLDAvis.sklearn
import warnings
import os
import nltk
import gensim
from gensim import corpora
from django.http import HttpResponse
from django.template import loader 


def ldarun(request):

#############     Initialise Data set   ###########
    path = "C:/Users/Admin/Desktop/Topic_Modelling/FileUpload/django-upload-example/media/books/pdfs"
    doc_set = []

    for file_name in os.listdir(path):
        if ".txt" in file_name:
            with open(os.path.join(path,file_name), "r") as src_file:
                data = src_file.read()
                doc_set.append(data)
        else:
            continue

	#print(len(doc_set))


	#############     Pipelining   ###########


    nltk.download('stopwords')
    nltk.download('wordnet')

    warnings.filterwarnings(action='ignore', category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning) 



    stop = set(stopwords.words('english'))
    exclude = set(string.punctuation) 
    lemma = WordNetLemmatizer()
    def clean(doc):
        stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
        punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
        normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
        return normalized

    doc_filtered = [clean(doc).split() for doc in doc_set]



	#############     Create Document-Term Matrix   ###########

	# Create id for each term. 
    dictionary = corpora.Dictionary(doc_filtered)

	# Create DTM from dictionary.
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_filtered]



	#############     Topic Modelling Algorith (LDA for now. extend to lda2vec)   ###########
							
	# Create LDA
    Lda = gensim.models.ldamodel.LdaModel

	# Run LDA.
    ntopics = request.POST.get("ntopics", None)
    ldamodel = Lda(doc_term_matrix, num_topics= ntopics, id2word = dictionary, passes=100)


	#############     Output (extend to pyLDAvis)   ###########

	#print(ldamodel)
	
    pyldavis = pyLDAvis.gensim.prepare(ldamodel, doc_term_matrix, dictionary)

    pyLDAvis.save_html(pyldavis, 'C:/Users/Admin/Desktop/Topic_Modelling/FileUpload/django-upload-example/mysite/templates/LDA/lda.html')
	
    template = loader.get_template('C:/Users/Admin/Desktop/Topic_Modelling/FileUpload/django-upload-example/mysite/templates/LDA/lda.html')
    return HttpResponse(template.render())
	
	

