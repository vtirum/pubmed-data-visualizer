import edirect_impl as e
import numpy as np
import pandas as pd
import sklearn.feature_extraction
import sklearn.decomposition
import streamlit as st
import random


__all__ = [
    "fetch_and_analyze", 
    "preprocess_dataframe", 
    "run_topic_modeling", 
    "compute_topic_stats",
]

@st.cache_data()
def fetch_and_analyze(query, start_year, end_year, n_topics=6): 
    pmids = e.pubmed_search(query, start_year, end_year)

    if len(pmids) == 0:
        return None, None, None
    
    SAMPLE_SIZE = 1500
    if len(pmids) > SAMPLE_SIZE:
        pmids = random.sample(pmids, SAMPLE_SIZE)
    
    df_raw = e.pubmed_request(pmids)
    df = preprocess_dataframe(df_raw)
    df, topic_labels = run_topic_modeling(df, n_topics)
    topic_dist, topic_trends = compute_topic_stats(df)

    return topic_dist, topic_trends, topic_labels


def preprocess_dataframe(df_raw):
    df = df_raw.copy()
    df["title"] = df["title"].map(e.normalize_whitespace)
    df["abstract"] = df["abstract"].map(e.normalize_whitespace)

    df["text"] = (
        df["title"].fillna("") + ". " + df["abstract"].fillna("")
    ).map(e.preprocess_text)

    df = df[(df["text"].str.len() > 20) & df["year"].notna()].copy()
    df["year"] = df["year"].astype(int)
    df.reset_index(drop=True, inplace=True)

    return df


def run_topic_modeling(df, n_topics):
    vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(
        max_features=10000,
        min_df=5,
        max_df=0.9,
        ngram_range=(1, 2),
    )

    X = vectorizer.fit_transform(df["text"])

    nmf = sklearn.decomposition.NMF(
        n_components=n_topics,
        random_state=42,
        init="nndsvda",
        max_iter=400,
    )

    W = nmf.fit_transform(X)
    H = nmf.components_
    df["topic"] = W.argmax(axis=1)
    feature_names = np.array(vectorizer.get_feature_names_out())

    def top_words(topic_idx, topn=6):
        idx = H[topic_idx].argsort()[-topn:][::-1]
        return ", ".join(feature_names[idx])

    topic_labels = {k: top_words(k) for k in range(n_topics)}
    df["topic_label"] = df["topic"].map(topic_labels)

    return df, topic_labels


def compute_topic_stats(df):
    topic_dist = df["topic_label"].value_counts()

    counts = (
        df.groupby(["year", "topic_label"])
        .size()
        .rename("n")
        .reset_index()
    )

    totals = counts.groupby("year")["n"].transform("sum")
    counts["share"] = counts["n"] / totals

    pivot_share = counts.pivot(
        index="year",
        columns="topic_label",
        values="share"
    ).fillna(0)

    return topic_dist, pivot_share


