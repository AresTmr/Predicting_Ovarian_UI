from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prototype.streamlit_app.candidate_response_service import (
    candidate_response_available,
    current_efficacy_run_id,
    current_run_id,
    score_candidate_response,
)
from prototype.streamlit_app.dose_recommendation_service import (
    predict_ui_reduced_dose_context,
    ui_reduced_model_available,
)
from prototype.streamlit_app.layer1_action_inference_service import (
    patient_form_to_snapshot,
    predict_layer1_action_context,
)
from prototype.streamlit_app.ui_real_data_sources import (
    build_knn_drug_summary,
    load_phase867_dose_shap_summary,
    load_phase867_local_dose_shap_for_patient,
    load_phase870_dose_probability_baselines,
)


def main() -> None:
    case = json.loads(Path("prototype/streamlit_app/representative_case.json").read_text(encoding="utf-8"))
    form = {}
    form.update(case.get("patient", {}))
    records = case.get("monitoring_records", [])
    if records:
        form.update(records[-1])
        form["monitoring_records"] = records
    snapshot = patient_form_to_snapshot(form)

    dose_available, dose_error = ui_reduced_model_available()
    if not dose_available:
        raise RuntimeError(f"dose model unavailable: {dose_error}")
    dose = predict_ui_reduced_dose_context(form)
    dose_labels = {drug: dose["predictions"][drug]["display"] for drug in ("fsh", "lh", "hmg")}

    candidate_available, candidate_error = candidate_response_available()
    if not candidate_available:
        raise RuntimeError(f"candidate-response unavailable: {candidate_error}")
    candidate = score_candidate_response(snapshot, fsh_dose=200, lh_dose=75, hmg_dose=150)

    shap_summary = load_phase867_dose_shap_summary(limit=3)
    local_shap = load_phase867_local_dose_shap_for_patient(form, limit=3)
    background = load_phase870_dose_probability_baselines()
    if sorted(shap_summary.get("targets", {})) != ["fsh", "hmg", "lh"]:
        raise RuntimeError("global dose SHAP targets incomplete")
    if sorted(local_shap.get("targets", {})) != ["fsh", "hmg", "lh"]:
        raise RuntimeError("local dose SHAP targets incomplete")
    if sorted(background.get("baselines", {})) != ["fsh", "hmg", "lh"]:
        raise RuntimeError("dose background probabilities incomplete")

    knn_context = predict_layer1_action_context(snapshot, k=50)
    knn_summary = build_knn_drug_summary(knn_context, limit=3)
    if {drug: len(rows) for drug, rows in knn_summary.items()} != {"fsh": 3, "lh": 3, "hmg": 3}:
        raise RuntimeError("KNN drug summaries incomplete")

    print("OK dose_labels", dose_labels)
    print("OK candidate_runs", current_efficacy_run_id(), current_run_id())
    print("OK candidate_oocytes_ohss", round(candidate["oocytes"], 3), round(candidate["ohss_risk"], 5))
    print("OK knn_cases", len(knn_context.get("similar_cases", [])))


if __name__ == "__main__":
    main()
