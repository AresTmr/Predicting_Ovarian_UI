# Predicting Ovarian UI

Standalone Streamlit prototype for the IVF/ICSI GnRH-a ultra-long protocol Gn dose decision-support UI.

## Setup steps

Recommended Python: 3.10 or 3.11. Do not use an unpinned Python 3.12 base environment directly, because some of the bundled model artifacts were produced with scikit-learn 1.3.x and a Python 3.12 environment may run into model pickle compatibility issues.

### 1. Download the code

Recommended: clone with Git so you can edit and commit later:

```bash
git clone https://github.com/Haaan1011/Predicting_Ovarian_UI.git
cd Predicting_Ovarian_UI
```

If you only want to look at the UI, you can also click `Code` -> `Download ZIP` on the GitHub page and unzip it into the project folder.

### 2. Create an environment and install dependencies

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you already have the original project environment on your machine, you can activate it directly:

```bash
conda activate Han_Overian
```

### 3. Launch the interface

```bash
streamlit run prototype/streamlit_app/app.py --server.address 127.0.0.1 --server.port 18501
```

Open the following URL in your browser after launch:

```text
http://127.0.0.1:18501/?view=knn
```

Common page URLs:

- Patient Input page: `http://127.0.0.1:18501/?view=input`
- Monitoring Results page: `http://127.0.0.1:18501/?view=monitor`
- Decision Curve page: `http://127.0.0.1:18501/?view=knn`
- Recommendation Explanation page: `http://127.0.0.1:18501/?view=shap`

### 4. Modify the UI code

The main files to edit are:

```text
prototype/streamlit_app/app.py
prototype/streamlit_app/ui_real_data_sources.py
prototype/streamlit_app/candidate_response_service.py
prototype/streamlit_app/dose_recommendation_service.py
```

For layout, copy, or style tweaks, prefer editing `prototype/streamlit_app/app.py`. Do not modify the main paper project, training data, paper result tables, or model training scripts.

### 5. Self-check after editing

```bash
python -m py_compile prototype/streamlit_app/app.py prototype/streamlit_app/ui_real_data_sources.py prototype/streamlit_app/dose_recommendation_service.py prototype/streamlit_app/candidate_response_service.py
python scripts/qa/smoke_check_ui_models.py
```

Launch Streamlit to view the UI after the self-check passes.

### 6. Commit your changes

```bash
git status
git add prototype/streamlit_app/app.py prototype/streamlit_app/ui_real_data_sources.py
git commit -m "update UI"
git push
```

If you only downloaded the ZIP instead of `git clone`, you can edit and run locally, but you cannot `git push` back to GitHub.

## What is included

- Streamlit clinician UI: patient input, monitoring result, decision curves, recommendation explanation.
- Current model integration needed by the UI: UI-reduced GRU(AddGate) dose models, candidate oocyte/OHSS response bundles, local SHAP support tables, and KNN support artifacts.
- De-identified runtime tables required for KNN/background statistics.

## What is intentionally not included

- Raw source spreadsheets or direct patient identifiers.
- Manuscript drafts, paper figure workflows, and unrelated training outputs.
- Downstream outcome UI outputs that are outside the current prototype scope. The prototype currently displays predicted oocyte yield and strict moderate-to-severe OHSS risk only.

## Smoke check

```bash
python scripts/qa/smoke_check_ui_models.py
```

## Clinical boundary

This is a clinical decision-support prototype, not an automatic order-entry system. The dose output should be described as a model-recommended dose or candidate-dose scenario analysis, and final medication decisions remain with the clinician. SHAP is model attribution; KNN is similar-case evidence; dose-response curves are conditional scenario predictions, not causal effects.