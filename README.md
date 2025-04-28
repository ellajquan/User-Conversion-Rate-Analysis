# User-Conversion-Rate-Analysis
Analyse user-level data, understand drivers of conversion, and build a Random-Forest model that predicts conversion rate so marketing can target high-propensity audiences.
## Key questions

1. What does the user profile look like?  
2. Which profile attributes correlate with conversion?  
3. Top indicators of high / low conversion?  
4. How does a segment with the *highest* propensity behave?  
5. Who should we target in a new marketing campaign?  
6. Model choice, implementation, and evaluation.  
7. Post-model actions – how do we *raise* conversion using the insights?

---

## Tech stack

| Layer | Tools |
|-------|-------|
| Data wrangling | **Python, pandas** |
| Visual EDA | **Matplotlib / Seaborn** |
| Modelling | **scikit-learn RandomForestClassifier** |
| Extras | Notebook → HTML report inside `docs/` (served via GitHub Pages) |

---

## Repository layout

```text
conversion-rate-project3/
├── data/                 # raw & processed CSVs (git-ignored large files)
├── notebooks/
│   ├── 01_eda.ipynb      # descriptive analysis, plots
│   └── 02_model.ipynb    # random-forest build & evaluation
├── src/
│   ├── prepare_data.py   # cleaning / feature engineering
│   └── train_rf.py       # CLI: python src/train_rf.py --config configs/default.yml
├── docs/                 # rendered report for GitHub Pages
│   └── index.md
├── requirements.txt
└── README.md
🚀 Quick start
bash
Copy
Edit
# 1️⃣ clone + install
git clone https://github.com/<user>/conversion-rate-project3.git
cd conversion-rate-project3
pip install -r requirements.txt     # or: conda env create -f env.yml

# 2️⃣ clean & feature-engineer
python src/prepare_data.py          # outputs data/processed.csv

# 3️⃣ train model
python src/train_rf.py              # writes models/random_forest.pkl

# 4️⃣ view report
open docs/index.html                # or visit the GitHub Page URL
📊 Results (example numbers)

Metric	Train	Test
Accuracy	0.87	0.84
ROC-AUC	0.93	0.90
Top 5 features	avg_session_time, pages_viewed, age, recent_purchase, mobile_os	
Users in the 90ᵗʰ-percentile propensity bucket exhibit a 3× higher conversion rate; ideal for retargeting ads.


