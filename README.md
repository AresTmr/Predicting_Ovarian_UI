# Predicting Ovarian UI

Standalone Streamlit prototype for the IVF/ICSI GnRH-a ultra-long protocol Gn dose decision-support UI.

## What is included

- Streamlit clinician UI: patient input, monitoring result, decision curves, recommendation explanation.
- Current model integration needed by the UI: UI-reduced GRU(AddGate) dose models, candidate oocyte/OHSS response bundles, local SHAP support tables, and KNN support artifacts.
- De-identified runtime tables required for KNN/background statistics.

## What is intentionally not included

- Raw source spreadsheets or direct patient identifiers.
- Manuscript drafts, paper figure workflows, and unrelated training outputs.
- Downstream outcome UI outputs that are outside the current prototype scope. The prototype currently displays predicted oocytes and strict moderate-to-severe OHSS risk only.

## Run

Use the project conda environment if available:

`ash
conda activate Han_Overian
streamlit run prototype/streamlit_app/app.py --server.address 127.0.0.1 --server.port 18501
`

Or create a fresh environment and install dependencies:

`ash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run prototype/streamlit_app/app.py --server.address 127.0.0.1 --server.port 18501
`

Open: http://127.0.0.1:18501/?view=knn

## Smoke check

```bash
python scripts/qa/smoke_check_ui_models.py
```

## Clinical boundary

This is a clinical decision-support prototype, not an automatic order-entry system. The dose output should be described as model-recommended dose or candidate-dose scenario analysis, and final medication decisions remain with the clinician. SHAP is model attribution; KNN is similar-case evidence; dose-response curves are conditional scenario predictions, not causal effects.
