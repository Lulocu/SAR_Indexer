import json
from nltk.stem.snowball import SnowballStemmer
import os
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
import numpy as np
import numpy.linalg as LA
import pickle
import random
import sys
import math


class SAR_Project:
    """
    Prototipo de la clase para realizar la indexacion y la recuperacion de noticias
        Preparada para todas las ampliaciones:
          parentesis + multiples indices + posicionales + stemming + permuterm + ranking de resultado
    Se deben completar los metodos que se indica.
    Se pueden añadir nuevas variables y nuevos metodos
    Los metodos que se añadan se deberan documentar en el codigo y explicar en la memoria
    """

    # lista de campos, el booleano indica si se debe tokenizar el campo
    # NECESARIO PARA LA AMPLIACION MULTIFIELD
    fields = [("title", True), ("date", False),
              ("keywords", True), ("article", True),
              ("summary", True)]


    # numero maximo de documento a mostrar cuando self.show_all es False
    SHOW_MAX = 10


    def __init__(self):
        """
        Constructor de la classe SAR_Indexer.
        NECESARIO PARA LA VERSION MINIMA
        Incluye todas las variables necesaria para todas las ampliaciones.
        Puedes añadir más variables si las necesitas
        """
        self.index = {} # hash para el indice invertido de terminos --> clave: termino, valor: posting list.
                        # Si se hace la implementacion multifield, se pude hacer un segundo nivel de hashing de tal forma que:
                        # self.index['title'] seria el indice invertido del campo 'title'.
        self.sindex = {} # hash para el indice invertido de stems --> clave: stem, valor: lista con los terminos que tienen ese stem
        self.ptindex = {} # hash para el indice permuterm.
        self.docs = {} # diccionario de terminos --> clave: entero(docid),  valor: ruta del fichero.
        self.weight = {} # hash de terminos para el pesado, ranking de resultados. puede no utilizarse
        self.news = {} # hash de noticias --> clave entero (newid), valor: la info necesaria para diferencia la noticia dentro de su fichero
        self.tokenizer = re.compile("\W+") # expresion regular para hacer la tokenizacion
        self.stemmer = SnowballStemmer('spanish') # stemmer en castellano
        self.show_all = False # valor por defecto, se cambia con self.set_showall()
        self.show_snippet = False # valor por defecto, se cambia con self.set_snippet()
        self.use_stemming = False # valor por defecto, se cambia con self.set_stemming()
        self.use_ranking = False  # valor por defecto, se cambia con self.set_ranking()

        #######VARIABLES NUEVAS#######
        self.title = {} # diccionario para almacenar los tokens de los titulos
        self.dates = {} # diccionario para almacenar las fechas
        self.keywords = {} # diccionario para almacenar los tokens de los keywords
        self.article = {} # diccionario para almacenar los tokens de article
        self.summary = {} # diccionario para almacenar los tokens de summary

        self.pttitle = {} # diccionario para almacenar los tokens de los titulos
        self.ptdates = {} # diccionario para almacenar las fechas
        self.ptkeywords = {} # diccionario para almacenar los tokens de los keywords
        self.ptarticle = {} # diccionario para almacenar los tokens de article
        self.ptsummary = {} # diccionario para almacenar los tokens de summary

        self.stitle = {} # diccionario para almacenar los tokens de los titulos
        self.sdates = {} # diccionario para almacenar las fechas
        self.skeywords = {} # diccionario para almacenar los tokens de los keywords
        self.sarticle = {} # diccionario para almacenar los tokens de article
        self.ssummary = {} # diccionario para almacenar los tokens de summary
        self.articulos = {} #diccionario de articulos para cada noticia

        self.idDoc = 0
        self.idNew = 0

    ###############################
    ###                         ###
    ###      CONFIGURACION      ###
    ###                         ###
    ###############################


    def set_showall(self, v):
        """
        Cambia el modo de mostrar los resultados.
        input: "v" booleano.
        UTIL PARA TODAS LAS VERSIONES
        si self.show_all es True se mostraran todos los resultados el lugar de un maximo de self.SHOW_MAX, no aplicable a la opcion -C
        """
        self.show_all = v


    def set_snippet(self, v):
        """
        Cambia el modo de mostrar snippet.
        input: "v" booleano.
        UTIL PARA TODAS LAS VERSIONES
        si self.show_snippet es True se mostrara un snippet de cada noticia, no aplicable a la opcion -C
        """
        self.show_snippet = v


    def set_stemming(self, v):
        """
        Cambia el modo de stemming por defecto.
        input: "v" booleano.
        UTIL PARA LA VERSION CON STEMMING
        si self.use_stemming es True las consultas se resolveran aplicando stemming por defecto.
        """
        self.use_stemming = v


    def set_ranking(self, v):
        """
        Cambia el modo de ranking por defecto.
        input: "v" booleano.
        UTIL PARA LA VERSION CON RANKING DE NOTICIAS
        si self.use_ranking es True las consultas se mostraran ordenadas, no aplicable a la opcion -C
        """
        self.use_ranking = v




    ###############################
    ###                         ###
    ###   PARTE 1: INDEXACION   ###
    ###                         ###
    ###############################


    def index_dir(self, root, **args):
        """
        NECESARIO PARA TODAS LAS VERSIONES
        Recorre recursivamente el directorio "root"  y indexa su contenido
        los argumentos adicionales "**args" solo son necesarios para las funcionalidades ampliadas
        """
        print("ARGUMENTOS:" + str(args) )
        self.multifield = args['multifield']
        self.positional = args['positional']
        self.stemming = args['stem']
        self.permuterm = args['permuterm']

        for dir, subdirs, files in os.walk(root):
            #print("DIR:" + dir)
            for filename in files:
                if filename.endswith('.json'):
                    fullname = os.path.join(dir, filename)
                    self.index_file(fullname)

        ##########################################
        ## COMPLETAR PARA FUNCIONALIDADES EXTRA ##
        ##########################################


    def index_file(self, filename):
        """
        NECESARIO PARA TODAS LAS VERSIONES
        Indexa el contenido de un fichero.
        Para tokenizar la noticia se debe llamar a "self.tokenize"
        Dependiendo del valor de "self.multifield" y "self.positional" se debe ampliar el indexado.
        En estos casos, se recomienda crear nuevos metodos para hacer mas sencilla la implementacion
        input: "filename" es el nombre de un fichero en formato JSON Arrays (https://www.w3schools.com/js/js_json_arrays.asp).
                Una vez parseado con json.load tendremos una lista de diccionarios, cada diccionario se corresponde a una noticia
        """
        #print("Nombre fichero: " + filename)
        with open(filename) as fh:
            jlist = json.load(fh)

        #
        # "jlist" es una lista con tantos elementos como noticias hay en el fichero,
        # cada noticia es un diccionario con los campos:
        #      "title", "date", "keywords", "article", "summary"
        #
        # En la version basica solo se debe indexar el contenido "article"
        #print("SE VA A INTRODUCIR TODOS LOS TOKEN EN index")
        self.docs[self.idDoc] = filename
        for noticia in jlist:
            ##################ARTICLE################################################
            tokens = self.tokenize(noticia['article'])
            self.articulos[self.idNew] = noticia['article']
            numToken = 0
            for token in tokens:
                tokenAux = token
                if self.index.get(token) == None:
                    self.index[token] = [[self.idDoc,self.idNew,numToken]]
                    self.article[tokenAux] = [[self.idDoc,self.idNew,numToken]]
                else:
                    aux = self.index.get(token)
                    aux.append([self.idDoc,self.idNew,numToken])
                    self.index[token] = aux

                    aux = self.article.get(tokenAux)
                    aux.append([self.idDoc,self.idNew,numToken])
                    self.article[tokenAux] = aux
                numToken += 1
            self.news[self.idNew] = [noticia["title"],noticia["date"],noticia["keywords"],noticia["summary"]]

            if self.multifield == True:
                ##################TITLE################################################
                tokensTitle = self.tokenize(noticia['title'])
                numToken = 0
                for token in tokensTitle:
                    tokenAux = token
                    token = "title:" + token

                    if self.index.get(token) == None:
                        self.index[token] = [[self.idDoc,self.idNew,numToken]]
                        self.title[tokenAux] = [[self.idDoc,self.idNew,numToken]]
                    else:
                        aux = self.index.get(token)
                        aux.append([self.idDoc,self.idNew,numToken])
                        self.index[token] = aux

                        aux = self.title.get(tokenAux)
                        aux.append([self.idDoc,self.idNew,numToken])
                        self.title[tokenAux] = aux
                    numToken += 1

                ##################SUMMARY################################################
                tokensSummary = self.tokenize(noticia['summary'])
                numToken = 0
                for token in tokensSummary:
                    tokenAux = token
                    token = "summary:" + token
                    if self.index.get(token) == None:
                        self.index[token] = [[self.idDoc,self.idNew,numToken]]
                        self.summary[tokenAux] = [[self.idDoc,self.idNew,numToken]]
                    else:
                        aux = self.index.get(token)
                        aux.append([self.idDoc,self.idNew,numToken])
                        self.index[token] = aux

                        aux = self.summary.get(tokenAux)
                        aux.append([self.idDoc,self.idNew,numToken])
                        self.summary[tokenAux] = aux
                    numToken += 1

                ##################KEYWORDS################################################
                tokensKeywords = self.tokenize(noticia['keywords'])
                numToken = 0
                for token in tokensKeywords:
                    tokenAux = token
                    token = "keywords:" + token
                    if self.index.get(token) == None:
                        self.index[token] = [[self.idDoc,self.idNew,numToken]]
                        self.keywords[tokenAux] = [[self.idDoc,self.idNew,numToken]]
                    else:
                        aux = self.index.get(token)
                        aux.append([self.idDoc,self.idNew,numToken])
                        self.index[token] = aux

                        aux = self.keywords.get(tokenAux)
                        aux.append([self.idDoc,self.idNew,numToken])
                        self.keywords[tokenAux] = aux
                    numToken += 1

            ##################DATE################################################
            if self.dates.get(noticia["date"]) == None:
                self.dates[noticia["date"]] = 1
            else:
                self.dates[noticia["date"]] = self.dates.get(noticia["date"]) + 1

            self.idNew += 1
        self.idDoc += 1

        #print("SE VA A INTRODUCIR TODOS LOS TOKEN PERMUTERM EN ptindex")
        #self.make_permuterm()
        #print(self.index)
        #print(len(self.title))




    def tokenize(self, text):
        """
        NECESARIO PARA TODAS LAS VERSIONES
        Tokeniza la cadena "texto" eliminando simbolos no alfanumericos y dividientola por espacios.
        Puedes utilizar la expresion regular 'self.tokenizer'.
        params: 'text': texto a tokenizar
        return: lista de tokens
        """
        return self.tokenizer.sub(' ', text.lower()).split()



    def make_stemming(self):
        """
        NECESARIO PARA LA AMPLIACION DE STEMMING.
        Crea el indice de stemming (self.sindex) para los terminos de todos los indices.
        self.stemmer.stem(token) devuelve el stem del token
        """
        if(self.multifield != True):
            for token in self.index.keys():
                stem = self.stemmer.stem(token)
                if self.sindex.get(stem) == None:
                    self.sindex[stem] = [token]
                else:
                    aux = self.sindex.get(stem)
                    aux.append(token)
                    self.sindex[stem] = aux
        else:
            for token in self.title.keys():
                stem = self.stemmer.stem(token)
                if self.stitle.get(stem) == None:
                    self.stitle[stem] = [token]
                else:
                    aux = self.stitle.get(stem)
                    aux.append(token)
                    self.stitle[stem] = aux

            for token in self.dates.keys():
                stem = self.stemmer.stem(token)

                if self.sdates.get(stem) == None:
                    self.sdates[stem] = [token]
                else:
                    aux = self.sdates.get(stem)
                    aux.append(token)
                    self.sdates[stem] = aux

            for token in self.keywords.keys():
                stem = self.stemmer.stem(token)

                if self.skeywords.get(stem) == None:
                    self.skeywords[stem] = [token]
                else:
                    aux = self.skeywords.get(stem)
                    aux.append(token)
                    self.skeywords[stem] = aux

            for token in self.article.keys():
                stem = self.stemmer.stem(token)

                if self.sarticle.get(stem) == None:
                    self.sarticle[stem] = [token]
                else:
                    aux = self.sarticle.get(stem)
                    aux.append(token)
                    self.sarticle[stem] = aux

            for token in self.summary.keys():
                stem = self.stemmer.stem(token)

                if self.ssummary.get(stem) == None:
                    self.ssummary[stem] = [token]
                else:
                    aux = self.ssummary.get(stem)
                    aux.append(token)
                    self.ssummary[stem] = aux



    def make_permuterm(self):
        """
        NECESARIO PARA LA AMPLIACION DE PERMUTERM
        Crea el indice permuterm (self.ptindex) para los terminos de todos los indices.
        """

        for token in self.index.keys():
            aux = token + '$'

            for i in range(len(aux)):
                aux = aux[1:] + aux[0]

                if self.ptindex.get(aux) == None:
                    self.ptindex[aux] = [token]
                else:
                    aux2 = self.ptindex.get(aux)
                    aux2.append(token)
                    self.ptindex[aux] = aux2

        if self.permuterm == True:
            for token in self.title.keys():
                aux = token + '$'

                for i in range(len(aux)):
                    aux = aux[1:] + aux[0]

                    if self.pttitle.get(aux) == None:
                        self.pttitle[aux] = [token]
                    else:
                        aux2 = self.pttitle.get(aux)
                        aux2.append(token)
                        self.pttitle[aux] = aux2

            for token in self.dates.keys():
                aux = token + '$'

                for i in range(len(aux)):
                    aux = aux[1:] + aux[0]

                    if self.ptdates.get(aux) == None:
                        self.ptdates[aux] = [token]
                    else:
                        aux2 = self.ptdates.get(aux)
                        aux2.append(token)
                        self.ptdates[aux] = aux2

            for token in self.keywords.keys():
                aux = token + '$'

                for i in range(len(aux)):
                    aux = aux[1:] + aux[0]

                    if self.ptkeywords.get(aux) == None:
                        self.ptkeywords[aux] = [token]
                    else:
                        aux2 = self.ptkeywords.get(aux)
                        aux2.append(token)
                        self.ptkeywords[aux] = aux2

            for token in self.article.keys():
                aux = token + '$'

                for i in range(len(aux)):
                    aux = aux[1:] + aux[0]

                    if self.ptarticle.get(aux) == None:
                        self.ptarticle[aux] = [token]
                    else:
                        aux2 = self.ptarticle.get(aux)
                        aux2.append(token)
                        self.ptarticle[aux] = aux2

            for token in self.summary.keys():
                aux = token + '$'

                for i in range(len(aux)):
                    aux = aux[1:] + aux[0]

                    if self.ptsummary.get(aux) == None:
                        self.ptsummary[aux] = [token]
                    else:
                        aux2 = self.ptsummary.get(aux)
                        aux2.append(token)
                        self.ptsummary[aux] = aux2




    def show_stats(self):
        """
        NECESARIO PARA TODAS LAS VERSIONES
        Muestra estadisticas de los indices
        """
        print("=" * 40)
        if self.multifield == True:
            print("Number of indexed days: " + str(len(self.dates)))
            print("-" * 40)
            print("Number of indexed news: " + str(len(self.news)))
            print("-" * 40)
            print("TOKENS:")
            print("\t# tokens in 'title': " + str(len(self.title)))
            print("\t# tokens in 'date': " + str(len(self.dates)))
            print("\t# tokens in 'keywords': " + str(len(self.keywords)))
            print("\t# tokens in 'article': " + str(len(self.article)))
            print("\t# tokens in 'summary': " + str(len(self.summary)))
            print("-" * 40)
        else:
            print("Number of indexed days: " + str(len(self.dates)))
            print("-" * 40)
            print("Number of indexed news: " + str(len(self.news)))
            print("-" * 40)
            print("TOKENS: " + str(len(self.index)))
            print("-" * 40)
            print("Positional queries are NOT allowed.")
            print("-" * 40)

        if self.permuterm == True:
            self.make_permuterm()
            if self.multifield == True:
                print("PERMUTERMS:")
                print("\t# permuterms in 'title': " + str(len(self.pttitle)))
                print("\t# permuterms in 'date': " + str(len(self.ptdates)))
                print("\t# permuterms in 'keywords': " + str(len(self.ptkeywords)))
                print("\t# permuterms in 'article': " + str(len(self.ptarticle)))
                print("\t# permuterms in 'summary': " + str(len(self.ptsummary)))
                print("-" * 40)
            else:
                print("PERMUTERMS:" + str(len(self.ptindex)))
                print("-" * 40)

        if self.stemming == True:
            self.make_stemming()
            if self.multifield == True:
                print("STEMS:")
                print("\t# stems in 'title': " + str(len(self.stitle)))
                print("\t# stems in 'date': " + str(len(self.sdates)))
                print("\t# stems in 'keywords': " + str(len(self.skeywords)))
                print("\t# stems in 'article': " + str(len(self.sarticle)))
                print("\t# stems in 'summary': " + str(len(self.ssummary)))
                print("-" * 40)
            else:
                print("STEMS:" + str(len(self.sindex)))
                print("-" * 40)

        if self.positional == True:
            print("Positional queries are allowed.")
        else:
            print("Positional queries are NOT allowed.")
        print("=" * 40)


    ###################################
    ###                             ###
    ###   PARTE 2.1: RECUPERACION   ###
    ###                             ###
    ###################################


    def solve_query(self, query, prev={}):
        """
        NECESARIO PARA TODAS LAS VERSIONES
        Resuelve una query.
        Debe realizar el parsing de consulta que sera mas o menos complicado en funcion de la ampliacion que se implementen
        param:  "query": cadena con la query
                "prev": incluido por si se quiere hacer una version recursiva. No es necesario utilizarlo.
        return: posting list con el resultado de la query
        """
        res = []
        listaPosting = []
        nPar = -1

        if query is None or len(query) == 0:
            return []

        consultaPartes = query.split()
        #print("Consulta partes: ",consultaPartes)

        j = 0

        while j <len(consultaPartes):
            palabra = consultaPartes[j]
            if palabra not in {'OR','NOT','AND','(OR','(AND','(NOT','OR)','NOT)','AND)'}:
                palabra = palabra.lower()

            #Miramos si hay paréntesis
            if palabra[0] == '(':
                nPar = 1
                #Miro si hay más '('
                x = 1
                while x < len(palabra):
                    if palabra[x] == '(':
                        nPar = nPar + 1
                        x = x + 1 
                    else:
                        x = len(palabra)
                #fin mirar


                i = j +1
                
                #Para todas las palabras de la consulta hasta cerrar el paréntesis
                while nPar > 0 and i < len(consultaPartes):
                    parFinal = consultaPartes[i]

                    #si la palabra empieza por '(' y si tiene más paréntesis
                    if parFinal[0] == '(':
                        nPar = nPar + 1
                        x = 1
                        while x < len(parFinal):
                            if parFinal[x] == '(':
                                nPar = nPar + 1
                                x = x+1
                            else:
                                x = len(parFinal)

                    #si la palabra termina en ')' y si tiene más paréntesis
                    if parFinal[-1] == ')':
                        nPar = nPar -1
                        x = len(parFinal) -2

                        while x >= 0 and nPar > 0:
                            if parFinal[x] == ')':
                                nPar = nPar - 1
                                x = x - 1
                            else:
                                x = -1
                    i = i +1

                medio = [palabra[1:]] + consultaPartes[j+1:i -1] + [parFinal[0:-1]]
                medio = ' '.join(medio)
                listaPosting.append(self.solve_query(medio))
                j = i -1
            

            #Miramos si hay parte posicional
            elif palabra[0] == '"':
                for parFinal in reversed(consultaPartes):
                    if parFinal[-1] == '"':
                        indFin = consultaPartes.index(parFinal)
                        medio =consultaPartes[j:indFin +1]
                        #Llamamos a este método pero sin paréntesis
                        listaPosting.append(self.get_posting(medio))
                        j = indFin

            elif palabra in {'AND','OR','NOT'}:
                listaPosting.append(palabra)

            else:
                listaPosting.append(self.get_posting(palabra))
            j = j + 1

        #print(listaPosting)
        #En listaPosting tenemos todas las posting list y los operadores binarios
        if listaPosting[0] == 'NOT':
            res = self.reverse_posting(listaPosting[1])
            i = 2
        else:
            res = listaPosting[0]
            i = 1

        while i < len(listaPosting):
            if listaPosting[i] == 'AND':
                i = i + 1
                if listaPosting[i] == 'NOT':
                    i = i + 1
                    res = self.and_posting(res,self.reverse_posting(listaPosting[i]))
                else:
                    res = self.and_posting(res,listaPosting[i])

            if listaPosting[i] == 'OR':
                i = i + 1
                if listaPosting[i] == 'NOT':
                    i = i + 1
                    res = self.or_posting(res,self.reverse_posting(listaPosting[i]))
                else:

                    res = self.or_posting(res,listaPosting[i])

            i = i + 1

        return res


        ########################################
        ## COMPLETAR PARA TODAS LAS VERSIONES ##
        ########################################




    def get_posting(self, term, field='article'):
        """
        NECESARIO PARA TODAS LAS VERSIONES
        Devuelve la posting list asociada a un termino.
        Dependiendo de las ampliaciones implementadas "get_posting" puede llamar a:
            - self.get_positionals: para la ampliacion de posicionales
            - self.get_permuterm: para la ampliacion de permuterms
            - self.get_stemming: para la amplaicion de stemming
        param:  "term": termino del que se debe recuperar la posting list.
                "field": campo sobre el que se debe recuperar la posting list, solo necesario se se hace la ampliacion de multiples indices
        return: posting list
        """
        #print("HOLAAAAA")
        #self.make_permuterm()
        #print(self.ptindex)

        if "*" in term or "?" in term:
            #print("HOLAAAA " + term)
            return self.get_permuterm(term)

        #self.get_stemming(term)
        if term[0] == '"':
            return self.get_positionals(term)


        #VERSION BASICA
        arrayIdNews = []
        if self.index.get(term) != None:
            postingList = self.index.get(term)
            for elemento in postingList:
                if elemento[1] not in arrayIdNews:
                        arrayIdNews.append(elemento[1])

        return arrayIdNews




    def get_positionals(self, terms, field='article'):
        """
        NECESARIO PARA LA AMPLIACION DE POSICIONALES
        Devuelve la posting list asociada a una secuencia de terminos consecutivos.
        param:  "terms": lista con los terminos consecutivos para recuperar la posting list.
                "field": campo sobre el que se debe recuperar la posting list, solo necesario se se hace la ampliacion de multiples indices
        return: posting list
        """
        print(self.positional)



    def get_stemming(self, term, field='article'):
        """
        NECESARIO PARA LA AMPLIACION DE STEMMING
        Devuelve la posting list asociada al stem de un termino.
        param:  "term": termino para recuperar la posting list de su stem.
                "field": campo sobre el que se debe recuperar la posting list, solo necesario se se hace la ampliacion de multiples indices
        return: posting list
        """

        stem = self.stemmer.stem(term)
        #self.make_stemming()

        return self.sindex.get(stem)




    def get_permuterm(self, term, field='article'):
        """
        NECESARIO PARA LA AMPLIACION DE PERMUTERM
        Devuelve la posting list asociada a un termino utilizando el indice permuterm.
        param:  "term": termino para recuperar la posting list, "term" incluye un comodin (* o ?).
                "field": campo sobre el que se debe recuperar la posting list, solo necesario se se hace la ampliacion de multiples indices
        return: posting list
        """
        aux = term + '$'

        #i = 0
        #for i in range(len(aux)):
        #    if aux[i] != '?' and aux[i] != '*':
        #        aux = aux[1:] + aux[0]
        #    else:
        #        aux = aux[1:] + aux[0]
        #        break
        #aux = aux[1:]

        alFinal = False
        while not alFinal:
            if aux[-1] == '?' or aux[-1] == '*':
                alFinal = True
            else:
                aux = aux[-1] + aux[0:-1]

        #print(self.ptindex.get(aux))
        print('AUX:')
        print(aux)
        print('Diccionario')
        for keys, values in self.ptindex.items():
            print(keys)
            print(values)
        return self.ptindex.get(aux)




    def reverse_posting(self, p):
        """
        NECESARIO PARA TODAS LAS VERSIONES
        Devuelve una posting list con todas las noticias excepto las contenidas en p.
        Util para resolver las queries con NOT.
        param:  "p": posting list
        return: posting list con todos los newid exceptos los contenidos en p
        """

        copiaDic = self.news.copy()
        claves = copiaDic.keys()
        poped=[]

        for i in p:
            if i in claves:
                poped.append(i)
                copiaDic.pop(i)
        #print("LAS QUE SE HAN BORRADO SON:")
        #print(poped)

        return list(copiaDic.keys())




    def and_posting(self, p1, p2):
        """
        NECESARIO PARA TODAS LAS VERSIONES
        Calcula el AND de dos posting list de forma EFICIENTE
        param:  "p1", "p2": posting lists sobre las que calcular
        return: posting list con los newid incluidos en p1 y p2
        """
        i = 0
        j = 0
        res = []

        while i < len(p1) and j < len(p2):

            if p1[i] == p2[j]:
                res.append(p1[i])
                i += 1
                j += 1
            else:
                if p1[i] < p2[j]:
                    i += 1
                else:
                    j += 1
        return res


    def or_posting(self, p1, p2):
        """
        NECESARIO PARA TODAS LAS VERSIONES
        Calcula el OR de dos posting list de forma EFICIENTE
        param:  "p1", "p2": posting lists sobre las que calcular
        return: posting list con los newid incluidos de p1 o p2
        """
        i = 0
        j = 0
        res = []

        while i < len(p1) and j < len(p2):
            if p1[i] == p2[j]:
                res.append(p1[i])
                i += 1
                j += 1
            else:
                if p1[i] < p2[j]:
                    res.append(p1[i])
                    i += 1
                else:
                    res.append(p2[j])
                    j += 1

        while i < len(p1):
            res.append(p1[i])
            i += 1

        while j < len(p2):
            res.append(p2[j])
            j += 1

        return res

    def minus_posting(self, p1, p2):
        """
        OPCIONAL PARA TODAS LAS VERSIONES
        Calcula el except de dos posting list de forma EFICIENTE.
        Esta funcion se propone por si os es util, no es necesario utilizarla.
        param:  "p1", "p2": posting lists sobre las que calcular
        return: posting list con los newid incluidos de p1 y no en p2
        """
        return and_posting(p1,reverse_posting(p2))




    #####################################
    ###                               ###
    ### PARTE 2.2: MOSTRAR RESULTADOS ###
    ###                               ###
    #####################################


    def solve_and_count(self, query):
        """
        NECESARIO PARA TODAS LAS VERSIONES
        Resuelve una consulta y la muestra junto al numero de resultados
        param:  "query": query que se debe resolver.
        return: el numero de noticias recuperadas, para la opcion -T
        """
        result = self.solve_query(query)
        #print("RESULTADO")
        #print(result)
        print("%s\t%d" % (query, len(result)))
        return len(result)  # para verificar los resultados (op: -T)


    def solve_and_show(self, query):
        """
        NECESARIO PARA TODAS LAS VERSIONES
        Resuelve una consulta y la muestra informacion de las noticias recuperadas.
        Consideraciones:
        - En funcion del valor de "self.show_snippet" se mostrara una informacion u otra.
        - Si se implementa la opcion de ranking y en funcion del valor de self.use_ranking debera llamar a self.rank_result
        param:  "query": query que se debe resolver.
        return: el numero de noticias recuperadas, para la opcion -T
        """
        result = self.solve_query(query)

        if self.use_ranking:
            ranking = self.rank_result(result, query)

        print("=" * 20)
        print("Query: '" + query + "'")
        print("Number of results: " + str(len(result)))

        i = 1
        for elemento in result:
            if i != 1:
                print("-" * 10)

            postingList = self.news.get(elemento)
            print("#" + str(i))
            print("Score: " + str(0))
            print(elemento)
            print("Date: " + postingList[1])
            print("Title: " + postingList[0])
            print("Keywords: " + postingList[2])
            if self.show_snippet:
                print("Summary: " + postingList[3])

            #print("-" * 10)
            i += 1
            if self.show_all == False and i > 100:
                break
        print("=" * 20)

    def snippet(self, noticiaID, query):

        path = self.docs[noticiaID]
        miNoticia = self.news[noticiaID]
        target = None
        tokens = []

        for token in query:
            if token != 'AND' and token != 'OR' and token != 'NOT':
                tokens.append(token)

        with open(path) as fh:
            jlist = json.load(fh)
            for noticia in jlist:
                if(noticia['title'] == miNoticia[0] and noticia['date'] == miNoticia):
                    target = noticia
                    break

    def rank_result(self, result, query):
        """
        NECESARIO PARA LA AMPLIACION DE RANKING
        Ordena los resultados de una query.
        param:  "result": lista de resultados sin ordenar
                "query": query, puede ser la query original, la query procesada o una lista de terminos
        return: la lista de resultados ordenada
        """
        
        news = []
        doc = []
        queryArray = []
        disCos = []
        for newsId in result:
                noticia = self.articulos[newsId]
                doc.append(noticia)

        #funcion para caclular distancia coseno
        distanciaCos = lambda a, b : round(np.inner(a, b)/(LA.norm(a)*LA.norm(b)), 3)

        stopWords = stopwords.words('spanish')
        vectorizer = CountVectorizer(stop_words = stopWords)
        transformer = TfidfTransformer()
        queryArray.append(query) 

        #Frequencias de documentos y queries
        vectorDocs = vectorizer.fit_transform(doc).toarray()
        vectorQuery = vectorizer.transform(queryArray).toarray()
        #transformer.fit(vectorQuery)
        #idfVectorQuery = transformer.transform(vectorQuery)
        #print idfVectorQuery.todense()

        for vector in vectorDocs:
            disCos.append(distanciaCos(vector, vectorQuery[0]))


        indices = np.array(disCos)
        indices = np.argsort(indices)
        indices = indices[::-1]

        res = []
        for i in indices:
            res.append(result[indices[i]])

        print(res)
        return [res]
