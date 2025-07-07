# Customer Review Classification with Google Gemini

This repository contains a Jupyter notebook—[**`data_classification_with_gemini.ipynb`**](data_classification_with_gemini.ipynb)—that demonstrates an end-to-end workflow for classifying customer reviews with **Google Gemini** (via LangChain). The example uses the *Disneyland Reviews* dataset from Kaggle and shows how to:

1. **Ingest & sample** a real-world review dataset.  
2. **Define a structured output schema** with Pydantic so an LLM can emit ready-to-use JSON.  
3. **Call Gemini** in batched loops with progress bars.  
4. **Flatten and analyse** the results in a Pandas DataFrame for downstream work such as dashboards or root-cause analysis.

---

## Notebook Outline

| Section | What it does |
|---------|--------------|
| **01 · Environment Setup** | Installs `langchain[google-genai]`, `pydantic`, and other Python packages. |
| **02 · Data Collection** | Authenticates to Kaggle and downloads the *Disneyland Reviews* CSV. |
| **03 · Data Preparation** | Loads the CSV to Pandas, samples 100 random rows, and partitions them into manageable batches. |
| **04 · LLM Initialisation** | Reads your **`GOOGLE_API_KEY`** from environment variables (or `google.colab.userdata` in Colab) and instantiates `gemini-2.5-flash-lite-preview-06-17` through LangChain. |
| **05 · Schema Definition** | Implements a `ReviewClassification` Pydantic model that tells the LLM exactly which fields to return. |
| **06 · Prompt Input Builder** | Converts each DataFrame batch to a fenced-CSV string (`df_to_prompt_input`) for injection into a LangChain prompt template. |
| **07 · LLM Execution** | Loops over the partitions, invokes Gemini, shows a custom-styled `tqdm` progress bar, and collects the responses. |
| **08 · Results Consolidation** | Normalises the structured output into a flat `df_processed` DataFrame with columns such as `review_id`, `summary`, `sentiment`, `priority_level`, `dominant_emotion`, `tags`, and `topics`. |

---

## Quick Start

> **Run on Colab**  
> Click the badge below to launch an interactive session with all dependencies pre-installed.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1GRfbLSRIJhiPM11hW0x7F1Q77e5CXgZ4?usp=sharing)

### 1 · Clone the repo

```bash
git clone https://github.com/Muzaiwan/text-analysis-with-gemini.git
cd text-analysis-with-gemini
```

### 2 · Install dependencies (local Jupyter)

```bash
python -m venv .venv && source .venv/bin/activate      # optional
pip install -U "langchain[google-genai]" pydantic pandas numpy tqdm kaggle chardet
```

### 3 · Set credentials

```bash
# Google Gemini
export GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"

# Kaggle (needed only the first time)
mkdir -p ~/.kaggle
cp /path/to/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 4 · Launch Jupyter Lab / Notebook

```bash
jupyter lab
```

Open **`data_classification_with_gemini.ipynb`** and run the cells top-to-bottom.

---

## Dataset

| Property | Value |
|----------|-------|
| **Name** | [Disneyland Reviews](https://www.kaggle.com/datasets/arushchillar/disneyland-reviews) |
| **Size** | ~21 k English reviews across three Disney parks |
| **License** | CC BY-NC-SA 4.0 (check Kaggle page for latest terms) |
| **Sampling** | Notebook randomly selects **100** rows (`random_state=42`) for quicker LLM experimentation. Adjust `df.sample(...)` if you need more data. |

---

## Output Schema (Pydantic)

```python
class ReviewClassification(BaseModel):
    review_id: int
    sentiment: Sentiment            # positive | neutral | negative
    priority_level: PriorityLevel   # low | medium | high
    summary: str                    # 1-sentence paraphrase
    sentiment_score: float          # −1.0 … 1.0
    dominant_emotion: Emotion       # enum, e.g. joy, anger
    tags: List[str]                 # 2–5 keywords
    topics: List[str]               # high-level topics
    language: str                   # ISO 639-1 code
```

The notebook auto-parses Gemini’s JSON output into this model and then into a tidy DataFrame.

---

## Example Result

| review_id | sentiment | priority_level | summary | sentiment_score |
|-----------|-----------|----------------|---------|-----------------|
| 10487 | negative | high | Long queues and overpriced food ruined the day. | –0.72 |
|  7123 | positive | low  | Loved the friendly staff and magical parades! | +0.85 |

*(Full table produced by `df_processed` in the final cell.)*

---

## Extending the Workflow

* **Hyperparameter sweeps** – Try different `temperature`, model sizes, or prompt templates.  
* **Topic modelling** – Feed `summary` and `topics` into clustering or BERTopic.  
* **Dashboarding** – Push `df_processed` to Looker Studio / Superset for live reports.  
* **Other datasets** – Replace the Kaggle download step with your own CSV of app reviews, Trustpilot feedback, etc.

---

## Troubleshooting & Tips

| Issue | Fix |
|-------|-----|
| `403: PERMISSION_DENIED` from Gemini | Double-check `GOOGLE_API_KEY`, quota, and model name. |
| `kaggle: command not found` | `pip install kaggle` or run inside Colab where it’s pre-installed. |
| Notebook stalls on large batches | Reduce `batch_size` in `np.array_split` or sample fewer rows. |

---

## References

* **LangChain Docs:** <https://python.langchain.com>  
* **Google Gemini Models:** <https://ai.google.dev/gemini-api/docs>  
* **Pydantic:** <https://docs.pydantic.dev>  
* **Disneyland Reviews Dataset:** Kaggle – arushchillar/disneyland-reviews  

---

## License

This project is released under the **MIT License** (see [`LICENSE`](LICENSE)). The Disneyland Reviews dataset remains subject to its original Kaggle license.

---

> Crafted with ❤️ by *Muhammad Zaidan Gunawan*. Contributions, issues, and suggestions are welcome!
