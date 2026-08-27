# Agent Notes

This repository is a UI handoff package only. Keep changes scoped to prototype/streamlit_app unless the UI needs a model-path or display-data fix. Do not add raw data, patient identifiers, manuscript drafts, transfer/embryo/pregnancy/live-birth UI outputs, or retraining artifacts.

Before handing back UI changes, run:

`ash
python -m py_compile prototype/streamlit_app/app.py prototype/streamlit_app/ui_real_data_sources.py prototype/streamlit_app/dose_recommendation_service.py prototype/streamlit_app/candidate_response_service.py
streamlit run prototype/streamlit_app/app.py --server.address 127.0.0.1 --server.port 18501
`
