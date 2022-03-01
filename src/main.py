#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NYU NLP Homework 4: Implement an ad hoc information retrieval system using TF-IDF
weights and cosine similarity scores.
    by Ziyang Zeng (zz2960)
    Spring 2022
"""


import argparse
from pathlib import Path
import math
from collections import Counter
import numpy as np
import nltk
from fastprogress.fastprogress import progress_bar

from typing import Dict, List, Tuple, Set, Union, TypedDict

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

closed_class_stop_words = ['a', 'the', 'an', 'and', 'or', 'but', 'about', 'above', 'after', 'along', 'amid', 'among',
                           'as', 'at', 'by', 'for', 'from', 'in', 'into', 'like', 'minus', 'near', 'of', 'off', 'on',
                           'onto', 'out', 'over', 'past', 'per', 'plus', 'since', 'till', 'to', 'under', 'until', 'up',
                           'via', 'vs', 'with', 'that', 'can', 'cannot', 'could', 'may', 'might', 'must',
                           'need', 'ought', 'shall', 'should', 'will', 'would', 'have', 'had', 'has', 'having', 'be',
                           'is', 'am', 'are', 'was', 'were', 'being', 'been', 'get', 'gets', 'got', 'gotten',
                           'getting', 'seem', 'seeming', 'seems', 'seemed',
                           'enough', 'both', 'all', 'your' 'those', 'this', 'these',
                           'their', 'the', 'that', 'some', 'our', 'no', 'neither', 'my',
                           'its', 'his' 'her', 'every', 'either', 'each', 'any', 'another',
                           'an', 'a', 'just', 'mere', 'such', 'merely' 'right', 'no', 'not',
                           'only', 'sheer', 'even', 'especially', 'namely', 'as', 'more',
                           'most', 'less' 'least', 'so', 'enough', 'too', 'pretty', 'quite',
                           'rather', 'somewhat', 'sufficiently' 'same', 'different', 'such',
                           'when', 'why', 'where', 'how', 'what', 'who', 'whom', 'which',
                           'whether', 'why', 'whose', 'if', 'anybody', 'anyone', 'anyplace',
                           'anything', 'anytime' 'anywhere', 'everybody', 'everyday',
                           'everyone', 'everyplace', 'everything' 'everywhere', 'whatever',
                           'whenever', 'whereever', 'whichever', 'whoever', 'whomever' 'he',
                           'him', 'his', 'her', 'she', 'it', 'they', 'them', 'its', 'their', 'theirs',
                           'you', 'your', 'yours', 'me', 'my', 'mine', 'I', 'we', 'us', 'much', 'and/or'
                           ]

stopwords = set([*nltk.corpus.stopwords.words('english'),
                *closed_class_stop_words])

stemmer = nltk.stem.snowball.EnglishStemmer()


def parse_data_file(file_path: Path) -> List[Dict[str, str]]:
    data = []
    last_data_item = None
    last_data_item_key = None
    last_data_item_value = None
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('.') and (len(line) == 3 or (len(line) > 3 and line[2] == ' ')):
                new_key = line[1]
                if last_data_item_key is not None:
                    last_data_item[last_data_item_key] = last_data_item_value.strip(
                    )
                if new_key == 'I':
                    if last_data_item is not None:
                        data.append(last_data_item)
                    last_data_item = {}
                last_data_item_key = new_key
                last_data_item_value = '' if len(line) == 2 else line[3:]
            else:
                last_data_item_value += line
        if last_data_item_key is not None:
            last_data_item[last_data_item_key] = last_data_item_value.strip()
        if last_data_item is not None:
            data.append(last_data_item)
    return data


def read_articles(file_path: Path, match_id: bool = False) -> List[str]:
    """
    Reads the articles from the file at the given path.
    :param file_path: The path to the file containing the articles.
    :return: A list of the articles.
    """
    data_list = parse_data_file(file_path)
    if match_id:
        for i, data_item in enumerate(data_list):
            if int(data_item['I']) != i + 1:
                print(
                    f'Error: I value {data_item["I"]} does not match index {i}')
    result: List[str] = []

    for data_item in data_list:
        item = ''
        if 'T' in data_item:
            item += data_item['T']
            item += ' '
        item += data_item['W']
        result.append(item)

    return result


def tokenize_article(article: str) -> List[str]:
    return [stemmer.stem(token.lower()) for token in nltk.tokenize.word_tokenize(article) if token.lower() not in stopwords and token.isalpha()]


def get_df_dict(articles: List[str]) -> Dict[str, int]:
    df_dict = {}
    for article in articles:
        token_set = set(tokenize_article(article))
        for token in token_set:
            if token not in df_dict:
                df_dict[token] = 0
            df_dict[token] += 1
    return df_dict


class VectorizationResult(TypedDict):
    vector: np.array
    tokens: List[str]


def vectorize_article(article: str, total_article_len: int, df_dict: Dict[str, int], tokens: List[str] = None) -> VectorizationResult:
    token_list = tokenize_article(article)
    if tokens is None:
        tokens = list(set(token_list))
    counter = Counter(token_list)
    vector = []
    for token in tokens:
        tf = counter[token] if token in counter else 0
        df = df_dict[token] if token in df_dict else 0
        idf = math.log(total_article_len / (df + 1))
        vector.append(tf * idf)
    return VectorizationResult(vector=np.array(vector), tokens=tokens)


def compute_cosine_similarity(v1: np.array, v2: np.array) -> float:
    if not v1.any() or not v2.any():
        return 0
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def main():
    parser = argparse.ArgumentParser(
        description="An ad hoc information retrieval system using TF-IDF weights and cosine similarity scores.")
    parser.add_argument("articlefile", help="input article corpus file")
    parser.add_argument("queryfile", help="input query file")
    parser.add_argument(
        "-o", dest="outputfile", help="path for storing output file, default is output.txt", default="output.txt")
    args = parser.parse_args()

    articles = read_articles(args.articlefile, match_id=True)
    total_article_len = len(articles)
    queries = read_articles(args.queryfile)

    df_dict = get_df_dict(articles)

    with open(args.outputfile, 'w') as f:
        query_index = 0
        for query in progress_bar(queries):
            query_index += 1
            query = vectorize_article(query, total_article_len, df_dict)
            article_index = 0
            query_result: List[Tuple[int, float]] = []
            for article in articles:
                article_index += 1
                article = vectorize_article(
                    article, total_article_len, df_dict, query['tokens'])
                cosine_similarity = compute_cosine_similarity(
                    query['vector'], article['vector'])
                if cosine_similarity > 0:
                    query_result.append((article_index, cosine_similarity))
            query_result.sort(key=lambda x: x[1], reverse=True)
            for result in query_result:
                f.write(f'{query_index} {result[0]} {result[1]}\n')

    print(f'Done. Output -> {args.outputfile}')


if __name__ == '__main__':
    main()
