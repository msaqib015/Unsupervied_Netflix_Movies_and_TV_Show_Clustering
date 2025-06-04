# 🎬 Unsupervised Netflix Movies and TV Show Clustering

This project applies **unsupervised machine learning techniques** to cluster Netflix's movies and TV shows based on their features such as genre, cast, release year, and description. The aim is to identify meaningful patterns and groupings that can help understand Netflix content better, potentially aiding in recommendation systems or content analysis.

## 🔍 Project Overview

* 📌 **Objective:** Perform exploratory data analysis (EDA) and cluster analysis on the Netflix dataset using unsupervised learning.
* 🧠 **Techniques Used:**

  * K-Means Clustering
  * Hierarchical Clustering
  * PCA for dimensionality reduction
  * TF-IDF vectorization for text data (e.g., descriptions)
* 📊 **Visualization Tools:** Matplotlib, Seaborn, Plotly

## 📁 Dataset

The dataset used is from [Netflix Movies and TV Shows on Kaggle](https://www.kaggle.com/datasets/shivamb/netflix-shows).

### Features include:

* Title
* Type (Movie/TV Show)
* Genre
* Cast
* Country
* Release Year
* Duration
* Description

## 🧪 Methods and Workflow

1. **Data Preprocessing**

   * Handling missing values
   * Encoding categorical variables
   * Text cleaning and vectorization

2. **Exploratory Data Analysis**

   * Insights into content distribution by genre, country, year, etc.
   * Word clouds and frequency analysis

3. **Clustering**

   * Feature engineering using TF-IDF and one-hot encoding
   * Dimensionality reduction using PCA
   * Applying K-Means and Hierarchical Clustering
   * Elbow method and Silhouette Score for optimal K selection

4. **Visualization**

   * Cluster visualization in 2D (via PCA)
   * Dendrograms for hierarchical clustering

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib, Seaborn, Plotly
* NLTK / spaCy (for NLP preprocessing)
* Jupyter Notebook

## 📌 Key Findings

* Clusters show strong grouping based on genre and description themes.
* Content can be grouped into broad categories like kids' shows, thrillers, romantic dramas, etc.
* Dimensionality reduction helped reveal latent structure in the content data.

## 📂 Repository Structure

```
Unsupervised_Netflix_Movies_and_TV_Show_Clustering/
│
├── data/                   # Dataset files (excluded in .gitignore if needed)
├── notebooks/              # Jupyter notebooks for EDA, clustering
├── visuals/                # Images and plots from the analysis
├── utils/                  # Utility functions
├── requirements.txt        # Python dependencies
└── README.md               # Project overview
```

## 🚀 Getting Started

### Prerequisites

Install the required packages using:

```bash
pip install -r requirements.txt
```

### Run the Analysis

Open the Jupyter notebook and run through the steps:

```bash
jupyter notebook notebooks/netflix_clustering.ipynb
```

## 🤝 Contributions

Contributions, suggestions, and feedback are welcome! Please open an issue or submit a pull request.

## 📜 License

This project is licensed under the MIT License.
