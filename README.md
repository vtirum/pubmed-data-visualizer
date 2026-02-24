# PubMed Data Visualizer

A Python tool for retrieving, analyzing, and visualizing topic trends in biomedical research using PubMed data.

This project queries PubMed records, processes publication metadata, and generates visualizations that show how research topics change over time.

---

## Features

- Fetches publication data from PubMed
- Extracts topic-related information from articles
- Computes topic distributions
- Tracks topic trends over time
- Generates publication-quality visualizations

---

## Example Outputs

### Topic Distribution
Shows the relative frequency of different research topics.

![Topic Distribution](topic_distribution.png)

### Topic Trends Over Time
Displays how topic shares change across years.

![Topic Trends](topic_shares_per_year.png)

---

## How It Works

The main script:
1. Queries PubMed using EDirect / API tools
2. Parses returned publication data
3. Aggregates topics
4. Produces visualizations using Python plotting libraries

---

## Requirements

Install dependencies:

```bash
pip install matplotlib pandas numpy


