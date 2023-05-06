# 1. Найти произведение матрицы A и обратной ей B и вывести все три матрицы на экран.
import numpy as np

A = np.array([[1, 2, 3], [2, 5, 1], [2, 3, 8]])
A_inv = np.linalg.inv(A)
C = np.dot(A, A_inv)
A1 = np.array([[0.5, 0.1], [1.7, 0.95]])
A1_inv = np.linalg.inv(A1)
C1 = np.dot(A1, A1_inv)
A2 = np.array([[1.1, 1, 1], [0, 1, 4.4], [2, 2.65, 2]])
A2_inv = np.linalg.inv(A2)
C2 = np.dot(A2, A2_inv)
print(C)
print(C1)
print(C2)

# 2. Найти решение СЛАУ
import numpy as np
from scipy.linalg import solve

A = np.array([[2, -5, 1], [1, 5, -4], [4, 1, -3]])
b = np.array([2, -5, -4]).reshape((3, 1))
x = solve(A, b)
print(x)

A = np.array([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]])
b = np.array([8, -11, -3]).reshape((3, 1))
x = solve(A, b)
print(x)

A = np.array([[1, 2, 3], [3, 5, 7], [1, 3, 4]])
b = np.array([3, 0, 1]).reshape((3, 1))
x = solve(A, b)
print(x)

# 5. Научиться работать с библиотекой natasha. Задачи:
# i.	Загрузить текстовые данные (не менее 2000 символов)
# ii.	Разделить текст на предложения
# iii.	Выделить токены и провести частеречную разметку, вывести на экран первые 20 токенов с тэгами
# iv.	Нормализовать именованные сущности в тексте
# v.	Выделить даты и вывести их в формате число-месяц-год
import natasha
import datetime
from datetime import datetime
from natasha import (Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger, NewsSyntaxParser, NewsNERTagger, PER,
NamesExtractor, DatesExtractor, Doc)

segmenter = Segmenter()
morph_vocab = MorphVocab()

emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)
ner_tagger = NewsNERTagger(emb)

names_extractor = NamesExtractor(morph_vocab)
dates_extractor = DatesExtractor(morph_vocab)

text = 'Центр фандрайзинга и работы с выпускниками СПбПУ провёл первый фестиваль «Выпускники Политеха — студентам»,
       'и оказалось, что, пожалуй, такого мероприятия давно не хватало. Лёгкий и неформальный вечер больше напоминал '
       'стендап, чем встречу с молодого поколения с наставниками. Конечно, профессионалам, довольно быстро и успешно '
       'после окончания вуза сделавшим карьеру, было что рассказать и чем поделиться с теми, кто ещё учится. Но '
       'немаловажно, что они сделали это с юмором, без нравоучительности и говорили с ребятами на одном языке. Тон '
       'задал выпускник Инженерно-экономического института 2012 года, проректор по молодёжной политике и '
       'коммуникативным технологиям СПбПУ Максим Пашоликов: Сегодня мы с вами проводим небольшой эксперимент. Мы хотим '
       'наладить коммуникацию студентов и выпускников, которые в разные годы окончили Политех, но не забывают его, '
       'готовы приезжать и общаться с молодёжью. Уже сейчас есть программы наставничества, стажировки, экскурсии на '
       'предприятия. Но мы хотим организовать прямой диалог между выпускниками конкретных кафедр, которые учились там '
       'же, где вы сейчас, общались с теми же преподавателями. В перспективе мы бы хотели, чтобы такие встречи '
       'проходили в каждом институте, высшей школе, где-то два раза в год. И нам очень интересно узнать ваше мнение '
       'об этой встрече, чтобы скорректировать работу в дальнейшем. После приветственных слов участники фестиваля '
       'разделились на две группы — технарей и гуманитариев. Впрочем, разделение было довольно условным, например, '
       'выпускник механико-машиностроительного факультета 2006 года Александр Леонов сейчас предприниматель, системный '
       'аналитик, разрабатывает информационные системы для управления бизнес-процессами и делает всё сам — от продажи '
       'IT-услуг до поддержки им же разработанного продукта. После фестиваля у студентов была ещё возможность '
       'пообщаться со спикерами лично. Один из слушателей технического трека, руководитель политеховского общественного '
       'института «Адаптеры», студент 4 курса Физмеха Александр Поталов поделился впечатлениями от встречи: Меня приятно'
       'удивило, как быстро ребята, которые в своё время выпустились из Политеха, смогли достичь того, чего они '
       'достигли. Я себе представлял этот путь гораздо более долгим. Очень здорово было увидеть, что люди из той сферы, '
       'в которой они изначально учились, двигались дальше, находили новые траектории развития. И если работают в своей '
       'отрасли, то развиваются с точки зрения проектов, предпринимательства. Очень крутая идея такого мероприятия. '
       'Мне, например, было бы интересно пообщаться с выпускниками моей кафедры, моего института. Так что я считаю, '
       'что это полезно, надо это развивать, чтобы в каждом институте такие встречи проходили.'
doc = Doc(text)

doc.segment(segmenter)
doc.tag_morph(morph_tagger)
print(doc.tokens[:20])
doc.tag_ner(ner_tagger)
for span in doc.spans:
    span.normalize(morph_vocab)
ner = {_.text: _.normal for _ in doc.spans}
print(ner)
dates = list(dates_extractor(text))
print(dates)

# 6. Средствами NLTK выделить именованные сущности с тэгами (Person, Organisation, GSP и проч.) для английского и
# русского текста.
import nltk
from nltk import word_tokenize
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

english_text = '''The Irish Grand National was run at Fairyhouse on Monday. The Mostly Irish Grand National, meanwhile,
 is at Aintree on Saturday, when 27 of the 40 runners facing the starter at 5.15pm will be attempting to extend
 Ireland’s current stranglehold on Britain’s most famous and popular race.'''

russian_text = '''Американский предприниматель Илон Маск создал компанию X.AI в области искусственного интеллекта,
зарегистрированную в штате Невада, сообщает газета Wall Street Journal со ссылкой на документы. "Согласно документам
штата, Илон Маск создал новую компанию в области искусственного интеллекта под названием X.AI, которая зарегистрирована
в Неваде", - говорится в сообщении газеты.'''

for sent in nltk.sent_tokenize(english_text):
   for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
      if hasattr(chunk, 'label'):
         print(chunk)

for sent in nltk.sent_tokenize(russian_text):
   for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
      if hasattr(chunk, 'label'):
         print(chunk)

# 7. С помощью sklearn обучить модель распознавать части речи в предложении. Для этого необходимо разбить данные на \
# обучающую и тестовую выборки, а в конце вывести на экран предсказание модели и степень его точности.
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

X_train = ["The word 'text' is a noun", "The word 'warm' is an adjective", "The word 'go' is a verb"]
y_train = ["noun", "adjective", "verb"]
X_test = ["The word 'text' is a noun", "The word 'go' is a verb"]

vectorizer = CountVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vectors, y_train)
predicted = model.predict(X_test_vectors)
print("Accuracy: ", accuracy_score(predicted, y_test))
