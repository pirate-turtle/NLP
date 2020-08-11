'''
1. colab에서 실행시 아래 명령어 실행
!apt-get update
!apt-get install g++ openjdk-8-jdk
!pip3 install konlpy
'''

from konlpy.tag import Mecab, Okt

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer

from collections import Counter
from wordcloud import WordCloud

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from PIL import Image



def get_morphs_kr(word_list):
    mecab = Mecab(dicpath="C:\\mecab\\mecab-ko-dic")
    # print(mecab.morphs(u'영등포구청역에 있는 맛집 좀 알려주세요.'))
    return mecab.morphs(word_list)


# 명사만 추출해서 불용어 제거하고 리턴
def get_clean_nouns_kr(word_list, stopwords):
    nouns = []
    tagger = Mecab(dicpath="C:\\mecab\\mecab-ko-dic")

    for word in word_list:
        for noun in tagger.nouns(word):
            if noun not in stopwords:
                nouns.append(noun)

    return nouns[0:10]


def get_clean_words_en(word_list, stopwords):
    # WordNet 사전에 들어간 단어를 기반으로 표제어추출(lemmatizing) 진행
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in word_list]

    # 불용어 제거하고 리턴
    clean_words = [word for word in lemmatized_words if word not in stopwords]

    return clean_words


# 문장을 형태소별로 나누고 토큰으로 분리하여 리턴
def get_token_kr(sentence):
    tokenizer = Okt()
    tokens = tokenizer.morphs(sentence)

    return tokens


def get_token_en(sentence):
    # + : 영문자, 숫자, 공백을 제외한 모든 문자를 제거한 뒤에 토큰화
    tokenizer = RegexpTokenizer(r'\w+')

    token = tokenizer.tokenize(sentence.lower())

    return token


# pos 태깅
def get_pos_tag_kr(word_list):
    tg_words = []
    tagger = Mecab(dicpath="C:\\mecab\\mecab-ko-dic")
    for word in word_list:
        tg_words.extend(tagger.pos(word))

    return tg_words


def get_noun_list_kr(malist):
    word_dic = {}
    # malist = [('사랑', 'Noun'), ('이', '조사'), ('사랑', 'Noun')]
    for word in malist:
        if word[1] == "Noun":  # 명사 확인하기 --- (※3)
            if not (word[0] in word_dic):
                word_dic[word[0]] = 0
            word_dic[word[0]] += 1  # 카운트하기
    keys = sorted(word_dic.items(), key=lambda x:x[1], reverse=True)

    return keys





def sort_by_keys(dict):
    return sorted(dict.keys(), reverse=True)


def sort_by_values(dict):
    return sorted(dict, key=dict.get, reverse=True)


def get_most_common_words(word_list, num):
    counter = Counter(word_list)
    top_words = dict(counter.most_common(num))

    return top_words


def draw_word_cloud(word_list, width=10, height=10, font_path = './font/malgun.ttf'):
    wc = WordCloud(background_color="white", font_path=font_path)
    wc.generate_from_frequencies(word_list)

    figure = plt.figure()
    figure.set_size_inches(width, height)
    ax = figure.add_subplot(1, 1, 1)
    ax.axis("off")
    ax.imshow(wc)


def draw_word_cloud_with_mask(word_list, image_path, font_path='./font/malgun.ttf', save=False, save_name='wordcloud.png'):
    mask = np.array(Image.open(image_path))

    # 워드 클라우드 설정
    mask_wc = WordCloud(background_color="white", mask=mask, contour_width=3,
                  font_path=font_path)


    mask_wc.generate_from_frequencies(word_list)

    # 이미지 표시
    plt.imshow(mask_wc, interpolation='bilinear')
    plt.axis("off")

    if save:
        # TODO 저장할 이름에 이미지 확장자 붙어있는지 확인. 없으면 붙여주기
        # 이미지 저장
        mask_wc.to_file(save_name)


def draw_bar_graph(word_list, barcount):
    font_location = 'c:/Windows/fonts/malgun.ttf'
    font_name = font_manager.FontProperties(fname=font_location).get_name()
    matplotlib.rc('font', family=font_name)

    plt.xlabel('주요 단어')
    plt.ylabel('빈도 수')
    plt.grid(True)

    Sorted_Dict_Values = sorted(word_list.values(), reverse=True)
    plt.bar(range(barcount), Sorted_Dict_Values[0:barcount], align='center')

    Sorted_Dict_Keys = sorted(word_list, key=word_list.get, reverse=True)
    plt.xticks(range(barcount), list(Sorted_Dict_Keys)[0:barcount], rotation='70')

    plt.show()


def tokens_to_bow(token_list, language='kr'):
    # 토큰을 문장으로 합침
    sentence = ' '.join(token_list)

    if language == 'kr':
        # 1글자도 인식이 되도록 토큰 패턴 변경
        cv = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
    else:
        cv = CountVectorizer()
    cv.fit(sentence)

    # CountVectorizer의 입력에 맞게 배열로 변경
    sentences = []
    sentences.append(sentence)

    # 벡터 변환
    vector = cv.transform(sentences).toarray()

    return vector


def draw_hist_by_token(token):
    plt.figure(figsize=(12, 5))

    plt.hist(token, bins=50, alpha=0.5, color='r', label='word')
    plt.yscale('log', nonposy='clip')

    plt.title('Review Length Histogram')

    plt.xlabel('Review Length')

    plt.ylabel('Number of Reviews')


def print_token_info(token):
    review_len_by_token = [len(t) for t in token]

    print('문장 최대 길이: {}'.format(np.max(review_len_by_token)))
    print('문장 최소 길이: {}'.format(np.min(review_len_by_token)))
    print('문장 평균 길이: {:.2f}'.format(np.mean(review_len_by_token)))
    print('문장 길이 표준편차: {:.2f}'.format(np.std(review_len_by_token)))
    print('문장 중간 길이: {}'.format(np.median(review_len_by_token)))
    print('제 1사분위 길이: {}'.format(np.percentile(review_len_by_token, 25)))
    print('제 3사분위 길이: {}'.format(np.percentile(review_len_by_token, 75)))


def draw_count_plot(dataframe, col_name):
    sentiment = dataframe[col_name].value_counts()
    fig, axe = plt.subplots(ncols=1)
    fig.set_size_inches(6, 3)
    sns.countplot(dataframe[col_name])