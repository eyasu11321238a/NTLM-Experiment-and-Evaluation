# <code>Thesis-Eyasu</code>

# <code>Investigating Transformer Input Embedding in Neural Translation Language Model for Search </code>

## Introduction

This thesis investigates effectiveness of the Neural Translation Language Model (NTLM) to address the vocabulary gap challenge in information retrieval. By integrating transformer-based input embeddings and Word2Vec models, it aims to enhance the semantic relationship between queries and documents for improved reranking effectiveness. The study uses the Dirichlet-smoothed language model as the initial ranker and evaluates the effectiveness of the NTLM against it. Additionally, we compare the NTLM with the WordLlama method to assess its effectiveness. Experiments on multiple datasets provide insights into the impact of different embeddings on ranking effectiveness.


## Research Questions

### 1. Comparative Analysis of Neural Translation Language Model

- Evaluates the effectiveness of the NTLM using Word2Vec embeddings and transformer input embeddings
- Compares against traditional Dirichlet-smoothed language model for document reranking using various datasets


### 2. Impact of Embedding Variations
- Investigates different types and configurations of Word2Vec models and transformer input embeddings
- Analyzes the relationship between embedding characteristics in word to word translation
- Examines the effectiveness of embedding from fine-tuned transformer model , like MonoBERT


### 3. Evaluating WordLlama's Effectiveness in NTLM
- How can WordLlama embeddings enhance NTLM effectiveness compared to Word2Vec and transformer input embeddings?

## Technical Framework

### Implementation Tools
- **PyTerrier Platform**: Used for first-stage retrieval, including:
  - Document indexing
  - Initial ranking
  - Evaluation
- Project Datasets repository: [Link to my webis repository]

### Datasets
- BEIR Datasets (https://github.com/beir-cellar/beir)
- ir_datasets (https://ir-datasets.com/)

### Evaluation Metrics
- Mean Average Precision (MAP)
- Precision at 10 (P@10)
- normalized Discounted Cumulative Gain (nDCG)


## Methodology


### Foundational Equation for NTLM Probabilistic Models for document ranking

The research methodology incorporates:

### 1. Traditional Probabilistic Models
#### Dirichlet-smoothed language model

- **Smoothed Probability**

The smoothed probability of a term $( w )$ given a document $( d )$ using Dirichlet smoothing is given by:

$$p_{\text{Dir}}(w|d) = \frac{c(w, d) + \mu \cdot p(w|C)}{|d| + \mu} \text{........................Eq(1) }$$

where:
- $c(w, d)$  is the count of term $( w )$ in document $( d )$.
- $|d|$  is the total number of terms in document $( d )$.
- $\mu$ is the Dirichlet smoothing parameter.
- $p(w|C)$ is the probability of term $( w )$ in the entire collection $( C )$.
###

- **Collection Probability**

The probability of term $( w )$ in the entire collection $( C )$ is:

$$p(w|C) = \frac{f(w, C)}{|C|}\text{........................Eq(2) }$$

where:
- $f(w, C)$ is the count or frequency of term $( w )$ in the entire collection $( C )$.
- $|C|$ is the total number of terms in the entire collection.


****



### 2. Neural Approaches
- Word2Vec embeddings
- Transformer input embeddings
- WordLlama
- Integration through translation language model framework



### Log-Likelihood for Document Ranking and NTLM method for Query Likelihood



1. **Log-Likelihood for Document Ranking**  
   Estimates the probability of generating the query \( q \) given a document \( d \).  

   $$
   \log p(q|d) = \sum_{i:c(q_i;d)>0} \log \left( \frac{p(q_i|d)}{\alpha_d \cdot p(q_i|C)} \right) + n \log \alpha_d \tag{Eq. 1}
   $$

   **Where:**  
   - $p(q_i|d)$: Probability of term \( q_i \) given document \( d \).  
   - $\alpha_d$: Normalizing factor.  
   - $p(q_i|C)$: Probability of term $q_i$ in the collection \( C \).  

2. **Cosine Similarity**  

   $$
   \text{cosine\_similarity}(u, w) = \frac{u \cdot w}{|u| \cdot |w|}
   $$

   $$
   p_t(w|u) = \frac{\text{cos}(u', w)}{\sum_{u_0 \in V} \text{cos}(u, w)}    \tag{Eq. 2 inserted into Eq. 3}
   $$

3. **Translation Model for Query Likelihood**  
   Models the process of query generation as a translation from document terms to query terms.  

   $$
   p_t(w|d) = \sum_{u \in d} p_t(w|u) \cdot p(u|d) \tag{Eq. 3}
   $$

   **Where:**  
   - \( p_t(w|u) \): Probability of translating document term \( u \) into query term \( w \).  
   - \( p(u|d) \): Probability of term \( u \) occurring in document \( d \).  

4. **Probability of Term \( u \) Occurring in Document \( d \):**  

   $$
   p(u|d) = \frac{\text{tf}(u, d)}{\sum_{v \in d} \text{tf}(v, d)} \tag{Eq. 4}
   $$

   **Where:**  
   - $\text{tf}(u, d)$: Frequency of term \( u \) in document \( d \).  
   - $\sum_{v \in d} \text{tf}(v, d)$: Total number of terms in document \( d \).  

5. **Probability of Query Term $q_i$ in the Collection $C$: $p(q_i|C)$**  

   The probability $p(q_i|C)$ represents the likelihood of term $q_i$ appearing in the entire collection $C$. It is calculated as the term frequency of $q_i$ in $C$ divided by the total number of terms in $C$.  

   $$
   p(q_i|C) = \frac{\text{cf}(q_i, C)}{\sum_{v \in C} \text{cf}(v, C)} \tag{Eq. 5}
   $$

   **Where:**  
   -  $\text{cf}(q_i, C)$: Collection frequency of term $q_i$ in $C$.  
   - $\sum_{v \in C} \text{cf}(v, C)$: Total number of terms in $C$.  

6. **Connecting the Two Concepts**  

   Specifically, $p(q_i|d)$ can be estimated using  $p_t(w|d)$:  

   $$
   p_t(w|d) \approx p_t(q_i|d)
   $$

   Substitute these equations into $\text{Eq. 1}$.  

7. **Final Log-Likelihood for Document Ranking Equation (Eq. 1):**  

   Using the Neural Translation Model, the log-likelihood equation becomes:  

   $$
   \log p(q|d) = \sum_{i:c(q_i;d)>0} \log \left( \frac{\sum_{u \in d} p_t(q_i|u) \cdot p(u|d)}{\alpha_d \cdot p(q_i|C)} \right) + n \log \alpha_d
   $$





















## Thesis Structure
```
Thesis_root/
│
├── document_ranking/
│   ├── __init__.py
│   ├── indexing.py
│   ├── scoring.py
│   ├── retrieval.py
│   ├── evaluation.py
│   ├── NTLM.ipynb
│
├── Dirichlet_model_Experiment/
│   ├──  
│   ├── 
│ 
│ 
├── NTLM_experiment/
│   ├──
│   ├──
│ 
│
└── data/
    ├── path_to_disk/
    ├── path_to_index/
    ├── path_to_topics/
    ├── path_to_qrels/
    ├── path_to_word_embeddings/
    └── path_to_trec_file/

```

## Installation

```bash
# Clone the repository
git clone https://git.webis.de/code-teaching/theses/thesis_eyasu.git

# Install dependencies
pip install -r requirements.txt

# Install PyTerrier
pip install python-terrier
```

## Usage

[how to run the code] 


