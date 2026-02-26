import streamlit as st
import matplotlib.pyplot as plt
from backend import fetch_and_analyze


st.title("PubMed Topic Trend Analyzer")

query = st.text_input("Enter PubMed search term")
years = st.slider("Select year range", 1990, 2026, (2010, 2020))

if st.button("Analyze"):
    with st.spinner("Fetching and analyzing data..."):
        dist, trends, labels = fetch_and_analyze(query, years[0], years[1])

    if dist is None:
        st.error("No results found")
    else:
        st.success("Analysis complete!")

        st.subheader("Topic Distribution")
        fig1, ax1 = plt.subplots()
        dist.plot(kind="bar", ax=ax1)
        st.pyplot(fig1)

        st.subheader("Topic Trends Over Time")
        fig2, ax2 = plt.subplots()
        trends.plot(ax=ax2)
        st.pyplot(fig2)