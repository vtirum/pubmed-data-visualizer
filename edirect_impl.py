import os, re, time, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from urllib.parse import quote_plus
from lxml import etree
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
import nltk
from nltk.corpus import stopwords
import subprocess
from concurrent.futures import ThreadPoolExecutor
from ratelimit import limits, sleep_and_retry


#nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))
plt.rcParams["figure.figsize"] = (10, 6)

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
API_KEY = os.getenv("NCBI_API_KEY", None)
EMAIL = "vtiru@umich.edu"

'''
def pubmed_search(query, start_year=None, end_year=None):
    \''' 
    Returns a list of pubmed IDs using Edirect matching the query in the date window
    \'''

    pmids = []
    
    esearch_cmd = f'esearch -db pubmed -query "{query}"'
    if start_year is not None and end_year is not None:
        esearch_cmd += f' -datetype PDAT -mindate {start_year} -maxdate {end_year}'

    efetch_cmd = f'efetch -format uid'
    cmd = f'{esearch_cmd} | {efetch_cmd}'

    result = subprocess.run(
        cmd, 
        capture_output=True, 
        text=True, 
        check=True, 
        shell=True
    )
    
    batch = result.stdout.splitlines()
    pmids.extend(batch)
    return pmids
'''

def pubmed_search(query, start_year=None, end_year=None, retmax=100000):
    """
    Search Pubmed via E-utilities and return a list of PMIDs
    """
    ids = []
    retstart = 0

    while True:
        params = {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retstart": retstart,
            "retmax": retmax,
            "email": EMAIL
        }
        if start_year and end_year:
            params["datetype"] = "pdat"
            params["mindate"] = start_year
            params["maxdate"] = end_year
        if API_KEY:
            params["api_key"] = API_KEY

        r = requests.get(f"{EUTILS}/esearch.fcgi", params=params)
        r.raise_for_status()
        data = r.json()

        batch_ids = data["esearchresult"].get("idlist", [])
        if not batch_ids:
            break
        ids.extend(batch_ids)

        retstart += retmax
        if retstart >= int(data["esearchresult"]["count"]):
            break

    return ids  

def chunker(seq, size):
    for pos in range(0, len(seq), size):
        yield seq[pos:pos+size]

def safe_text(elem):
    return (elem.text or "").strip() if elem is not None else ""

def parse_xml(xml_bytes):
    '''
    Returns a list of dicts:
    pmid, title, abstract, year, journal, mesh_terms (list), authors(list)
    '''
    
    root = etree.fromstring(xml_bytes)
    ns = {}
    records = []
    
    for article in root.findall(".//PubmedArticle", ns):
        pmid = safe_text(article.find(".//PMID"))
        title = safe_text(article.find(".//ArticleTitle"))

        # since abstracts can have multiple parts
        abstract_parts = []
        for abstract in article.findall((".//Abstract/AbstractText")):
            text = "".join(abstract.itertext()).strip()
            if text:
                abstract_parts.append(text)
        abstract = "".join(abstract_parts).strip()

        year = None
        y = safe_text(article.find(".//ArticleDate/Year"))
        if y.isdigit():
            year = int(y)
        else:
            y = safe_text(article.find(".//JournalIssue/PubDate/Year"))
            if y.isdigit():
                year = int(y)
            else:
                medline = safe_text(article.find(".//JournalIssue/PubDate/MedlineDate"))
                m = re.search(r"\b(19/20)\d{2}\b", medline)
                if m:
                    year = int(m.group())   

        journal = safe_text(article.find(".//JournalTitle"))

        mesh_terms = []
        for mh in article.findall(".//MeshHeading/DescriptorName"):
            term = "".join(mh.itertext()).strip()
            if term:
                mesh_terms.append(mh)

        authors = []
        for au in article.findall(".//Author"):
            fore = safe_text(au.find(".//ForeName"))
            last = safe_text(au.find(".//LastName"))
            aff = safe_text(au.find(".//Affiliation"))
            if aff is not None:
                authors.append(aff)
            elif last or fore:
                name = f"{fore} {last}".strip()
                authors.append(name)

        records.append({
            "pmid": pmid,
            "title": title, 
            "abstract": abstract, 
            "year": year, 
            "journal": journal, 
            "mesh_terms": mesh_terms, 
            "authors": authors
        })
    return records 


CALLS_PER_SEC = 1 #if not API_KEY else 8
@sleep_and_retry
@limits(calls=CALLS_PER_SEC, period=1)
def fetch_batch(batch):
    ids= ",".join(batch)
    params = {
        "db": "pubmed", 
        "rettype": "abstract",
        "retmode": "xml", 
        "id": ids,
        "api_key": API_KEY, 
        "email": EMAIL
    }
    r = requests.get(f"{EUTILS}/efetch.fcgi", params=params)
    r.raise_for_status()
    return parse_xml(r.content)


def pubmed_request(pmids, batch_size=200):
    """
    Fetch details from pubmed via efetch
    """
    batches = list(chunker(pmids, batch_size))
    all_rows = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        results = executor.map(fetch_batch, batches)

    for rows in results:
        all_rows.extend(rows)
    return pd.DataFrame(all_rows)

def normalize_whitespace(s):
    return re.sub(r"\s+", " ", s or "").strip()

def preprocess_text(s):
    s = (s or "").lower()
    s = re.sub(r"[^a-z\s]", " ", s)
    tokens = [w for w in s.split() if w not in STOPWORDS and len(w) > 2]
    return " ".join(tokens)





def main():
    TOPIC = "crispr"
    START_YEAR = 2015
    END_YEAR = 2016

    pmids = pubmed_search(TOPIC)
    print(f"Found {len(pmids)} PMIDS for query {TOPIC}")

    df_raw = pubmed_request(pmids)
    print(df_raw.shape)
    df_raw.head()

    df = df_raw.copy()
    df["title"] = df["title"].map(normalize_whitespace)
    df["abstract"] = df["abstract"].map(normalize_whitespace)
    df["text"] = (df["title"].fillna("") + ". " + df["abstract"].fillna("")).map(preprocess_text)

    df = df[(df["text"].str.len() > 20) & df["year"].notna()].copy()
    df["year"] = df["year"].astype(int)
    df.reset_index(drop=True, inplace=True)

    print(df.shape)
    df[["pmid", "year", "title", "abstract"]].head(3)

    N_TOPICS = 6
    MAX_FEATURES = 10000

    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES, 
        min_df=5, 
        max_df=0.9,
        ngram_range=(1, 2)
    )
    X = vectorizer.fit_transform(df["text"])

    nmf = NMF(
        n_components = N_TOPICS,
        random_state = 42,
        init = "nndsvda",
        max_iter = 400
    )
    W = nmf.fit_transform(X)
    H = nmf.components_

    topic_assign = W.argmax(axis=1)
    df["topic"] = topic_assign

    feature_names = np.array(vectorizer.get_feature_names_out())

    def top_words(H, topic_idx, topn=12):
        top_idx = H[topic_idx].argsort()[-topn:][::-1]
        return feature_names[top_idx]

    for k in range(N_TOPICS):
        print(f"topic {k}: {', '.join(top_words(H, k, 12))}")

    def auto_label_topic(k, topn=6):
        return ", ".join(top_words(H, k, topn))

    topic_labels = {k: auto_label_topic(k, 6) for k in range(N_TOPICS)}
    df["topic_label"] = df["topic"].map(topic_labels)
    topic_labels

    topic_counts = df["topic_label"].value_counts().sort_values(ascending=True)
    topic_counts.plot(kind="bar")
    plt.title("Topic distribution")
    plt.ylabel("Number of papers")
    plt.xlabel("Topic")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("topic_distribution.png")

    counts = (
        df.groupby(["year", "topic_label"])
        .size()
        .rename("n")
        .reset_index()
    )

    year_totals = counts.groupby("year")["n"].transform("sum")
    counts["share"] = counts["n"] / year_totals

    pivot_n = counts.pivot(index="year", columns="topic_label", values="n").fillna(0)
    pivot_share = counts.pivot(index="year", columns="topic_label", values="share").fillna(0)

    pivot_share.sort_index().plot()
    plt.title("Topic shares per year")
    plt.ylabel("Share")
    plt.xlabel("Year")
    plt.tight_layout()
    plt.savefig("topic_shares_per_year.png")


if __name__ == "__main__":
    main()