# 🤖 Tech Stack Recommender System

> **AI-powered career path recommendation engine** built using TF-IDF vectorization and Cosine Similarity — mapping your skills to the most relevant tech job roles with precision.

---

## 📸 Project Screenshots

### 🖥️ Step 1 — Enter Your Skills
![Enter Skills Interface](<img width="424" height="362" alt="prject-3" src="https://github.com/user-attachments/assets/71f71efd-1cab-4c72-9c27-d310ebda50d2" />


---

### 📊 Step 2 & 3 — Processing & All Roles Scored
![All Roles Scored](<img width="428" height="344" alt="proj-3" src="https://github.com/user-attachments/assets/ecff06b6-054e-40ff-9ef4-d7beab8ff218" />
)

---

### 🏆 Step 4 — Top Matches For You
![Top Matches](<img width="424" height="424" alt="prj-3" src="https://github.com/user-attachments/assets/964ad805-5a38-4d10-a1f1-6c6d7e56c2ce" />
)

---

### 📋 Summary Dashboard
![Summary](<img width="421" height="218" alt="project 3" src="https://github.com/user-attachments/assets/7702edbc-2e27-4576-9151-c706c6ec4e62" />
)

---

## 📌 Project Overview

The **Tech Stack Recommender System** is a content-based filtering engine that matches a user's technical skills to the most relevant job roles in tech. It implements a full **Input → Process → Output (IPO)** pipeline, using industry-standard NLP techniques to generate objective, ranked career recommendations.

Built as **Project 3** of the DecodeLabs AI Industrial Training Kit (Batch 2026).

---

## 🎯 Goal

> Create a simple but powerful recommendation system based on user preferences (skills), using pure algorithmic similarity logic — no external APIs, no collaborative data needed.

---

## ✨ Features

- 🔤 **Skill Input Interface** — Enter minimum 3 skills for accurate matching
- ⚙️ **TF-IDF Vectorization** — Weights specific skills higher than generic ones
- 📐 **Cosine Similarity Engine** — Measures directional alignment between user profile and job role vectors
- 📊 **Full Role Scoring Table** — Ranks all 15 job roles with match percentages and visual bars
- 🥇 **Top-3 Recommendations** — Displays the best matches with verdict labels (Strong Match / Good Match / Possible Match)
- 📋 **Summary Dashboard** — Clean overview of analyzed skills and ranked results

---

## 🧠 Algorithm & How It Works

This system follows a strict **4-step ranking pipeline**:

```
STEP 1: Ingestion   →   STEP 2: Scoring   →   STEP 3: Sorting   →   STEP 4: Filtering
   (User Input)        (Cosine Similarity)    (Descending Order)      (Top-N Output)
```

### 🔬 TF-IDF Weighting

| Component | Role |
|---|---|
| **Term Frequency (TF)** | Rewards terms that appear more in a specific job role |
| **Inverse Document Frequency (IDF)** | Penalizes generic terms like "software", "code" that appear everywhere |
| **TF × IDF** | Final weight — specific skills (e.g., "PyTorch") rank higher than generic ones |

### 📐 Cosine Similarity Formula

$$\cos(\theta) = \frac{A \cdot B}{\|A\| \|B\|}$$

| Score | Meaning |
|---|---|
| **1.0** | Perfect match — vectors point in the same direction |
| **0.0** | No overlap — orthogonal vectors |
| **-1.0** | Opposite — negative correlation |

---

## 🗂️ Job Roles Covered

| Rank | Job Role | Example Skills |
|---|---|---|
| ⭐ | Machine Learning Engineer | Python, TensorFlow, PyTorch, ML, Deep Learning |
| ⭐ | AI Research Scientist | Python, Math, Statistics, NLP, Computer Vision |
| ⭐ | Data Scientist | Python, SQL, Machine Learning, Data Analysis |
| | Full Stack Developer | React, Node.js, Python, Databases |
| | Backend Developer | Java, Python, APIs, SQL |
| | Cybersecurity Analyst | Networking, Linux, Cryptography |
| | Site Reliability Engineer | Docker, Kubernetes, CI/CD |
| | Blockchain Developer | Solidity, Web3, Cryptography |
| | Data Engineer | Spark, Airflow, SQL, ETL |
| | Frontend Developer | HTML, CSS, JavaScript, React |
| | DevOps Engineer | AWS, Docker, CI/CD, Ansible |
| | Mobile Developer | Flutter, React Native, Swift |
| | Cloud Architect | AWS, GCP, Azure, Terraform |
| | Database Administrator | SQL, PostgreSQL, MongoDB |
| | Systems Analyst | UML, Business Analysis, SQL |

---

## 🛠️ Tech Stack

```
Language    : Python 3.x
Algorithm   : TF-IDF + Cosine Similarity
Libraries   : scikit-learn, numpy, pandas
Interface   : Terminal / CLI
Dataset     : raw_skills.csv (job roles with skill tags)
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install scikit-learn numpy pandas
```

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/tech-stack-recommender.git

# Navigate into the project folder
cd tech-stack-recommender
```

### Run the System

```bash
python recommender.py
```

### Usage Example

```
STEP 1 — Enter Your Skills
Minimum 3 required for accurate matching

Skill 1: python
Skill 2: pytorch
Skill 3: machine learning
Skill 4: deep learning
Skill 5: tensorflow
Skill 6: scikit-learn

→ STEP 2: Processing & Scoring all roles...
→ STEP 3: Ranking results...
→ STEP 4: Top Matches For You:

  🥇 Machine Learning Engineer — 64.4% (Strong Match ✓)
  🥈 AI Research Scientist     — 52.4% (Good Match ✓)
  🥉 Data Scientist            — 33.3% (Possible Match)
```

---

## 📁 Project Structure

```
tech-stack-recommender/
│
├── recommender.py          # Main recommendation engine
├── raw_skills.csv          # Job roles dataset with skill tags
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## 📊 Sample Output

**Skills Analyzed:** `python` · `pytorch` · `machine learning` · `deep learning` · `tensorflow` · `scikit-learn`

| Rank | Job Role | Match % | Verdict |
|---|---|---|---|
| #1 | Machine Learning Engineer | **64.4%** | Strong Match ✓ |
| #2 | AI Research Scientist | **52.4%** | Good Match ✓ |
| #3 | Data Scientist | **33.3%** | Possible Match |
| #4 | Full Stack Developer | 4.9% | — |
| #5 | Backend Developer | 4.8% | — |

---

## 🧩 Key Concepts Demonstrated

- ✅ **Content-Based Filtering** — recommendations driven by item attributes, not user history
- ✅ **Vector Space Modeling** — user and job profiles mapped to a shared vocabulary space
- ✅ **TF-IDF Feature Extraction** — penalizes generic terms, rewards specific skill tags
- ✅ **Cosine Similarity** — magnitude-invariant similarity measurement (industry standard)
- ✅ **Cold Start Handling** — onboarding survey (minimum 3 skills) bootstraps the user vector
- ✅ **Top-N Filtering** — truncates output to prevent choice overload

---

## 📚 What I Learned

This project marked a fundamental shift from **passive classification** to **active prediction** — understanding how platforms like Netflix and Amazon match users to content before they even ask.

Key takeaways:
- Why Euclidean distance fails at scale (magnitude sensitivity)
- How TF-IDF solves the binary overlap limitation of simple tag matching
- The importance of a shared vocabulary space between user profiles and item datasets
- Cold start problem and how content-based filtering inherently bypasses the item cold start

---

## 🏗️ Built As Part Of

> **DecodeLabs Industrial Training Kit**
> Batch: 2026 | Domain: Artificial Intelligence
> Project 3 of the AI Track — *AI Recommendation Logic*

---

## 👤 Author

**Saad Mandli**

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<div align="center">
  <strong>Built with 🤖 by Saad Mandli · Algorithm: TF-IDF + Cosine Similarity</strong>
</div>
