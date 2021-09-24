#En esta script se generaran variables a partir de la metadata.

#Cargo datos
import pandas as pd
metadata = pd.read_json(r'C:\Users\argomezja\Desktop\Data Science\MELI challenge\items_static_metadata_full.jl', lines=True)


#%% Pre-procesing
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
metadata['titles_edited'] = metadata['item_title'].str.lower() 
metadata['titles_edited'] = remove_stop_words(zip(metadata.titles_edited, metadata.site_id))

metadata['titles_edited'] = stemming(zip(metadata.titles_edited, metadata.site_id))




#Tf-idf con los títulos de los productos
metadata_new = []
for site in metadata['site_id'].unique():
        subset = metadata.loc[metadata['site_id']==site].reset_index(drop = True)
        name_address =subset['titles_edited']
        vectorizer = TfidfVectorizer("char", ngram_range=(1, 4), sublinear_tf=True)
        tf_idf_matrix = vectorizer.fit_transform(name_address)
        print('calculo scores....')
        scores = tf_idf_matrix.sum(axis = 1)
        
        subset['suma_scores']= scores        
        scores = tf_idf_matrix.mean(axis = 1)    
        

        scores = tf_idf_matrix.max(axis = 1).toarray()
        subset['max_scores']= scores
        
        metadata_new.append(subset)
        print(site)
        
metadata =pd.concat(metadata_new)    

#Creo mas features extras
metadata['item_domain_id'] = metadata['item_domain_id'].str[4:]
metadata['item_domain_id'] = pd.factorize(metadata.item_domain_id)[0]
metadata['site_id'] = pd.factorize(metadata.site_id)[0]
metadata['totalwords'] = metadata['item_title'].str.split().str.len()
metadata['totalwords2'] = metadata['titles_edited'].str.split().str.len()
count = metadata['item_domain_id'].value_counts().reset_index()
metadata = metadata.merge(count, left_on = 'item_domain_id', right_on = 'index')

#Me quedo con lo que me sirve y lo guardo
extra = metadata[['sku', 'site_id', 'item_domain_id_y', 'totalwords','totalwords2','suma_scores', 'max_scores', 'item_domain_id_x']].copy()

extra.to_csv('features_extras.csv.gz',index=False, compression="gzip")






#%% Funciones
from nltk.stem import SnowballStemmer
from tqdm import tqdm
################################################################################
def stemming(columnas):

    spanish_stemmer = SnowballStemmer('spanish')
    portuguese_stemmer = SnowballStemmer('portuguese')
    stemming = []
    for title, web_page in  tqdm(columnas):
        if web_page=='MLB':
            stemming.append(' '.join([portuguese_stemmer.stem(x) for x in title.split()]))
        else:
            stemming.append(' '.join([spanish_stemmer.stem(x) for x in title.split()]))

    return stemming

################################################################################
def remove_stop_words(columnas):
    

    no_list_spanish = stopwords.words('spanish')
    no_list_port = stopwords.words('portuguese')
    no_stop_words = []
    for title, web_page in  tqdm(columnas):
        if web_page=='MLB':
             no_stop_words.append(' '.join([x for x in title.split() if x not in no_list_port]))
    
        else:
            no_stop_words.append(' '.join([x for x in title.split() if x not in no_list_spanish]))
    
    return no_stop_words