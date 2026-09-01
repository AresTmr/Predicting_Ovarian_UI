from __future__ import annotations
import csv
import json
import math
import os
import sys
from datetime import datetime, timedelta
from html import escape
from pathlib import Path
from typing import Any, Mapping
import streamlit as st

APP_ROOT = Path(__file__).resolve().parent
REPO_ROOT = APP_ROOT.parents[1]
for path in (APP_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
try:
    from layer1_action_inference_service import evidence_for_action, patient_form_to_snapshot, predict_layer1_action_context
    LAYER1_AVAILABLE = True
except Exception as exc:
    evidence_for_action = patient_form_to_snapshot = predict_layer1_action_context = None
    LAYER1_AVAILABLE = False
    LAYER1_ERROR = str(exc)
try:
    from candidate_scoring_service import score_effectiveness_safety_outcomes
    OUTCOME_AVAILABLE = True
except Exception as exc:
    score_effectiveness_safety_outcomes = None
    OUTCOME_AVAILABLE = False
    OUTCOME_ERROR = str(exc)
try:
    from candidate_response_service import (
        candidate_response_prediction_baselines,
        current_efficacy_run_id,
        current_run_id,
        explain_candidate_response,
        explain_candidate_response_shap,
        score_candidate_response,
    )
    CANDIDATE_RESPONSE_AVAILABLE = True
except Exception as exc:
    candidate_response_prediction_baselines = current_efficacy_run_id = current_run_id = explain_candidate_response = explain_candidate_response_shap = score_candidate_response = None
    CANDIDATE_RESPONSE_AVAILABLE = False
    CANDIDATE_RESPONSE_ERROR = str(exc)
try:
    from candidate_balance_recommendation_service import (
        apply_oocyte_ohss_balance_recommendation,
    )
    CANDIDATE_BALANCE_AVAILABLE = True
except Exception as exc:
    apply_oocyte_ohss_balance_recommendation = None
    CANDIDATE_BALANCE_AVAILABLE = False
    CANDIDATE_BALANCE_ERROR = str(exc)
try:
    from ohss_display_scaling import with_ohss_display_fields
except Exception:
    def with_ohss_display_fields(row):
        return dict(row or {})
try:
    from dose_recommendation_service import predict_ui_reduced_dose_context, ui_reduced_model_available
    DOSE_RECOMMENDATION_AVAILABLE = True
except Exception as exc:
    predict_ui_reduced_dose_context = ui_reduced_model_available = None
    DOSE_RECOMMENDATION_AVAILABLE = False
    DOSE_RECOMMENDATION_ERROR = str(exc)
try:
    from ui_real_data_sources import (
        UI_INPUT_ATTRIBUTION_FEATURES,
        build_knn_drug_summary,
        build_knn_reference_rows,
        build_current_patient_dose_attribution_items,
        load_phase867_dose_shap_summary,
        load_phase867_local_dose_shap_for_patient,
        load_phase870_dose_probability_baselines,
    )
    UI_REAL_DATA_AVAILABLE = True
except Exception as exc:
    UI_INPUT_ATTRIBUTION_FEATURES = set()
    build_current_patient_dose_attribution_items = build_knn_drug_summary = build_knn_reference_rows = load_phase867_dose_shap_summary = load_phase867_local_dose_shap_for_patient = load_phase870_dose_probability_baselines = None
    UI_REAL_DATA_AVAILABLE = False
    UI_REAL_DATA_ERROR = str(exc)

st.set_page_config(page_title="Gn 剂量辅助系统", page_icon="⚕", layout="wide", initial_sidebar_state="collapsed")
PAGES = ["患者录入", "监测结果", "决策曲线", "推荐解释"]
LEGACY = {"信息输入":"患者录入", "结果输出":"监测结果", "患者信息输入":"患者录入", "推荐方案":"监测结果", "KNN 曲线":"决策曲线", "recommend":"监测结果"}
PAGE_SLUGS = {"首页":"home", "患者录入":"input", "决策曲线":"knn", "推荐解释":"shap", "监测结果":"monitor"}
SLUG_PAGES = {v: k for k, v in PAGE_SLUGS.items()}
MAIN_LOCAL_SHAP_LIMIT = 6
KNN_CURVE_SENSITIVITY_GAIN = float(os.getenv("KNN_CURVE_SENSITIVITY_GAIN", "1.0"))
REPRESENTATIVE_CASE_PATH = APP_ROOT / "representative_case.json"

def load_representative_case(path=REPRESENTATIVE_CASE_PATH):
    try:
        payload=json.loads(Path(path).read_text(encoding="utf-8"))
        return payload if isinstance(payload, Mapping) else {}
    except Exception:
        return {}

REPRESENTATIVE_CASE = load_representative_case()
DEFAULT = dict(
    patient_id="Case-2026-014",
    age=32,
    bmi=21.2,
    years=3,
    infertility="继发不孕",
    diagnosis="继发不孕",
    amh=2.1,
    afc=12,
    basal_fsh=7.2,
    basal_lh=4.8,
    basal_e2=42.0,
    basal_p=0.5,
    protocol="GnRH-a 长方案",
    fertilization_method="IVF",
    male_factor_infertility="否",
    sperm_source_group="PESA/TESA",
    male_age=38,
    treatment_count=1,
    initial_gn=0.0,
    initial_fsh=0.0,
    initial_lh=0.0,
    initial_hmg=0.0,
    initial_gn_source="pending_first_monitoring_prediction",
    visit=1,
    stim_day=1,
    days_since_previous_visit=0,
    monitoring_date="2026-06-26",
    current_fsh=0.0,
    current_lh=0.0,
    current_hmg=0.0,
    e2=42.0,
    lh_value=4.8,
    p=0.50,
    serum_fsh=7.2,
    current_endometrium=5.6,
    total_follicles=12,
    left_follicles=6,
    right_follicles=6,
    f_lt10=12,
    f_10_12=0,
    f_13_15=0,
    f_16_18=0,
    f_gt18=0,
    max_f=8.5,
    mean_f=5.8,
)
if isinstance(REPRESENTATIVE_CASE.get("patient"), Mapping):
    DEFAULT.update(REPRESENTATIVE_CASE["patient"])
if REPRESENTATIVE_CASE.get("monitoring_records"):
    DEFAULT.update(dict(REPRESENTATIVE_CASE["monitoring_records"][0]))

OHSS_RISK_DEFAULTS = dict(
    threshold_low=float(os.getenv("OHSS_RISK_THRESHOLD_LOW", "0.05")),
    threshold_high=float(os.getenv("OHSS_RISK_THRESHOLD_HIGH", "0.10")),
    reference_predictions_path=os.getenv(
        "OHSS_RISK_REFERENCE_PREDICTIONS",
        "results/ohss_safety_warning/current/predictions.csv",
    ),
)
OHSS_UI_DISCLAIMER = "\u8be5\u7ed3\u679c\u4e3a\u4fc3\u6392\u9636\u6bb5\u4e2d\u91cd\u5ea6 OHSS \u98ce\u9669\u5206\u5c42\u548c\u5b89\u5168\u9884\u8b66\u53c2\u8003\uff0c\u4e0d\u4f5c\u4e3a\u8bca\u65ad\u7ed3\u8bba\u6216\u81ea\u52a8\u533b\u5631\u3002"
OHSS_FEATURE_LABELS = {
    "current_e2":"\u8840\u6e05 E2", "current_lh":"\u8840\u6e05 LH", "current_p":"\u8840\u6e05 P",
    "basal_e2":"\u57fa\u7840 E2", "basal_lh":"\u57fa\u7840 LH", "basal_fsh":"\u57fa\u7840 FSH", "basal_p":"\u57fa\u7840 P",
    "total_follicle_count":"\u603b\u5375\u6ce1\u6570", "mature_follicle_count":"\u226514 mm \u5375\u6ce1\u6570", "medium_plus_follicle_count":"\u226513 mm \u5375\u6ce1\u6570", "dominant_follicle_count":"\u4f18\u52bf\u5375\u6ce1\u6570",
    "follicle_count_lt_10":"<10 mm \u5375\u6ce1\u6570", "follicle_count_10_12":"10-12 mm \u5375\u6ce1\u6570", "follicle_count_gt_18":">18 mm \u5375\u6ce1\u6570", "follicle_count_16_18":"16-18 mm \u5375\u6ce1\u6570", "follicle_count_13_15":"13-15 mm \u5375\u6ce1\u6570", "growing_follicle_count":"\u751f\u957f\u5375\u6ce1\u6570",
    "left_follicle_count":"\u5de6\u4fa7\u5375\u6ce1\u6570", "right_follicle_count":"\u53f3\u4fa7\u5375\u6ce1\u6570",
    "max_follicle_diameter":"\u6700\u5927\u5375\u6ce1\u76f4\u5f84", "mean_follicle_diameter":"\u5e73\u5747\u5375\u6ce1\u76f4\u5f84",
    "current_gn_dose":"\u5f53\u524d\u603b Gn \u5242\u91cf", "current_fsh_daily_dose":"\u5f53\u524d FSH \u5242\u91cf", "current_lh_daily_dose":"\u5f53\u524d LH \u5242\u91cf", "current_hmg_daily_dose":"\u5f53\u524d HMG \u5242\u91cf", "current_lh_like_hmg_daily_dose":"\u5f53\u524d LH/HMG \u7c7b Gn \u5242\u91cf", "initial_gn_dose":"\u8d77\u59cb Gn \u5242\u91cf",
    "current_fsh":"\u8840\u6e05 FSH", "previous_fsh_daily_dose":"\u65e2\u5f80 FSH \u5242\u91cf", "previous_lh_daily_dose":"\u65e2\u5f80 LH \u5242\u91cf", "previous_hmg_daily_dose":"\u65e2\u5f80 HMG \u5242\u91cf", "days_since_previous_visit":"\u8ddd\u4e0a\u6b21\u76d1\u6d4b\u5929\u6570",
    "amh":"AMH", "afc":"AFC", "age":"\u5e74\u9f84", "bmi":"BMI", "gn_day":"\u4fc3\u6392\u65e5", "monitoring_order":"\u76d1\u6d4b\u6b21\u6570", "visits_seen":"\u5df2\u76d1\u6d4b\u6b21\u6570",
    "e2_per_mature_follicle":"\u5355\u4f4d\u6210\u719f\u5375\u6ce1 E2", "e2_per_weighted_follicle":"\u5355\u4f4d\u52a0\u6743\u5375\u6ce1 E2", "gn_per_total_follicle":"\u5355\u4f4d\u5375\u6ce1 Gn \u5242\u91cf", "current_e2_per_gn":"\u5355\u4f4d Gn E2", "large_follicle_share":"\u5927\u5375\u6ce1\u6bd4\u4f8b", "mature_follicle_share":"\u6210\u719f\u5375\u6ce1\u6bd4\u4f8b", "ohss_follicle_load_score":"\u5375\u6ce1\u8d1f\u8377\u8bc4\u5206", "p_lh_ratio":"P/LH \u6bd4\u503c", "infertility_duration":"\u4e0d\u5b55\u5e74\u9650", "current_endometrium":"\u5185\u819c\u539a\u5ea6",
}
OHSS_FEATURE_UNITS = {"current_e2":"pg/mL","basal_e2":"pg/mL","current_lh":"IU/L","basal_lh":"IU/L","current_p":"ng/mL","basal_p":"ng/mL","basal_fsh":"IU/L","current_fsh":"IU/L","max_follicle_diameter":"mm","mean_follicle_diameter":"mm","current_endometrium":"mm","current_gn_dose":"IU/\u5929","current_fsh_daily_dose":"IU/\u5929","current_lh_daily_dose":"IU/\u5929","current_hmg_daily_dose":"IU/\u5929","previous_fsh_daily_dose":"IU/\u5929","previous_lh_daily_dose":"IU/\u5929","previous_hmg_daily_dose":"IU/\u5929","current_lh_like_hmg_daily_dose":"IU/\u5929","initial_gn_dose":"IU/\u5929","age":"\u5c81","gn_day":"\u5929","days_since_previous_visit":"\u5929","monitoring_order":"\u6b21","visits_seen":"\u6b21","total_follicle_count":"\u4e2a","left_follicle_count":"\u4e2a","right_follicle_count":"\u4e2a","mature_follicle_count":"\u4e2a","medium_plus_follicle_count":"\u4e2a","dominant_follicle_count":"\u4e2a","follicle_count_lt_10":"\u4e2a","follicle_count_10_12":"\u4e2a","follicle_count_gt_18":"\u4e2a","follicle_count_16_18":"\u4e2a","follicle_count_13_15":"\u4e2a","growing_follicle_count":"\u4e2a"}
CSS = """
<style>
:root{--bg:#f8f9ff;--card:#fff;--ink:#0b1c30;--muted:#64748b;--line:#d8deeb;--p:#4f46e5;--pd:#3525cd;--ps:#eef0ff;--t:#14b8a6;--ts:#dcfbf6;--w:#f59e0b;--ws:#fff7ed;--d:#dc2626;--ds:#fff1f2;--fs-scale:1.5;--fs-scale-2x:2;--fs-scale-tight:1.2}
html,body,.stApp,[data-testid="stAppViewContainer"],[data-testid="stAppViewBlockContainer"]{background:var(--bg)!important;color:var(--ink)!important;color-scheme:light!important;font-family:"Microsoft YaHei UI","Microsoft YaHei","微软雅黑","PingFang SC",sans-serif!important}header,#MainMenu,footer{visibility:hidden;height:0}.block-container{max-width:1480px;padding:0 28px 82px!important;padding-top:0!important}
/* ===== 顶部导航栏：sticky 置顶、品牌靠左、页面标签整体水平居中、无边框 =====
   导航容器 = st.container() 生成的 stVerticalBlock（同时包含 .nav-brand 与 .st-key-page_selector）；
   品牌 markdown 绝对定位靠左，radio 占满整行后内部居中，四项整体居中。 */
div[data-testid="stVerticalBlock"]:has(.nav-brand):has(.st-key-page_selector){
  position:sticky!important;top:0!important;z-index:100!important;
  background:#fff!important;border:none!important;border-radius:0!important;
  box-shadow:none!important;margin:0 -28px!important;padding:0 28px!important
}
/* 清除容器内部 wrapper（stVerticalBlockBorderWrapper / padding 层）的多余留白 */
div[data-testid="stVerticalBlock"]:has(.nav-brand):has(.st-key-page_selector) [data-testid="stVerticalBlockBorderWrapper"]{
  padding:0!important;border:0!important;background:transparent!important;box-shadow:none!important
}
/* 最外层含导航的 stVerticalBlockBorderWrapper 也清 padding（消除顶部 15px 留白） */
div[data-testid="stVerticalBlockBorderWrapper"]:has([data-testid="stVerticalBlock"] .nav-brand){
  padding-top:0!important;border-top:0!important
}
div[data-testid="stVerticalBlock"]:has(.nav-brand):has(.st-key-page_selector) [data-testid="stVerticalBlock"]{
  padding:0!important;margin:0!important
}
/* 品牌：绝对定位靠左（sticky 容器建立定位上下文） */
div[data-testid="stVerticalBlock"]:has(.nav-brand):has(.st-key-page_selector) > .stElementContainer:first-child{
  position:absolute!important;left:0!important;top:0!important;height:56px!important;
  display:flex!important;align-items:center!important;z-index:6!important;
  width:auto!important;pointer-events:none!important;
  margin:0!important;padding:0!important;border:none!important;background:transparent!important
}
div[data-testid="stVerticalBlock"]:has(.nav-brand):has(.st-key-page_selector) > .stElementContainer:first-child .stMarkdownContainer{margin:0!important;padding:0!important;pointer-events:none!important}
.nav-brand{display:flex;align-items:center;gap:10px;height:56px;min-width:0;pointer-events:none!important}
.nav-brand .mark{width:30px;height:30px;border-radius:8px;background:#EFF4FF;color:#3B82F6;display:grid;place-items:center;font-weight:700;font-size:calc(16px * var(--fs-scale));flex:0 0 auto}
.nav-brand .brand-name{font-size:calc(18px * var(--fs-scale));font-weight:600;color:#1D2939;white-space:nowrap}
.nav-brand .sub{font-size:calc(12px * var(--fs-scale));color:#98A2B3;white-space:nowrap;margin-top:2px}
/* 标签（radio）：占满整行，四项整体水平居中 */
div[data-testid="stVerticalBlock"]:has(.nav-brand):has(.st-key-page_selector) .st-key-page_selector{
  width:100%!important;height:56px!important;display:flex!important;align-items:center!important;justify-content:center!important;margin:0!important;padding:0!important
}
div[data-testid="stVerticalBlock"]:has(.nav-brand):has(.st-key-page_selector) div[data-testid="stRadio"]{width:100%!important;padding:0!important}
div[data-testid="stVerticalBlock"]:has(.nav-brand):has(.st-key-page_selector) div[role="radiogroup"]{display:flex!important;justify-content:center!important;gap:2px!important;margin:0!important;position:relative!important;z-index:5!important;overflow:visible!important}
div[data-testid="stVerticalBlock"]:has(.nav-brand):has(.st-key-page_selector) div[role="radiogroup"] label{background:transparent!important;border:0!important;border-radius:0!important;padding:0 16px!important;height:56px!important;margin:0!important;display:flex!important;align-items:center!important;font-size:14px!important;font-weight:500!important;color:#475467!important;cursor:pointer!important;white-space:nowrap!important;transition:none!important}
div[data-testid="stVerticalBlock"]:has(.nav-brand):has(.st-key-page_selector) div[role="radiogroup"] label>div:first-child{display:none!important}
div[data-testid="stVerticalBlock"]:has(.nav-brand):has(.st-key-page_selector) div[role="radiogroup"] input{position:absolute!important;opacity:0!important;width:0!important;height:0!important}
div[data-testid="stVerticalBlock"]:has(.nav-brand):has(.st-key-page_selector) div[role="radiogroup"] label:hover{color:#1D2939!important}
div[data-testid="stVerticalBlock"]:has(.nav-brand):has(.st-key-page_selector) div[role="radiogroup"] label:has(input:checked){color:#1D2939!important;font-weight:600!important;box-shadow:inset 0 -2px 0 0 #3B82F6!important}
/* 隐藏注入的 <style> markdown 残留（无内容、避免占位导致导航下方偏移） */
div[data-testid="stMarkdown"]:has(style){display:none!important}
/* ===== 页面标题：24px/600/#1D2939，紧贴导航下方，副标题灰色小字 ===== */
.title{margin:18px 0 22px!important}.title h1{margin:0 0 6px!important;font-size:36px!important;line-height:1.4!important;font-weight:600!important;color:#1D2939!important}.title p{margin:0!important;color:#667085!important;font-size:13px!important;line-height:1.6!important}.card,.sec{background:var(--card);border:1px solid var(--line);border-radius:16px;box-shadow:0 4px 12px rgba(15,23,42,.04)}.pad{padding:24px}.sec{padding:22px 24px;margin-bottom:20px}.head{display:flex;align-items:center;gap:10px;margin-bottom:18px}.head .ic{width:28px;height:28px;border-radius:8px;background:var(--ps);color:var(--pd);display:grid;place-items:center;font-weight:900}.head h3{margin:0;font-size:calc(20px * var(--fs-scale))}.note{font-size:calc(12px * var(--fs-scale));color:var(--muted);line-height:1.55}.grid4{display:grid;grid-template-columns:repeat(4,1fr);gap:18px}.metric{background:#fff;border:1px solid var(--line);border-radius:16px;padding:20px}.ml{color:var(--muted);font-size:calc(13px * var(--fs-scale));font-weight:720}.mv{font-size:calc(30px * var(--fs-scale));font-weight:900;line-height:1.25;margin-top:8px;color:var(--pd)}.teal{color:#087d72!important}.warn{color:#d97706!important}.chip{display:inline-flex;border-radius:999px;padding:4px 10px;font-size:calc(12px * var(--fs-scale));font-weight:850;border:1px solid var(--line);background:#f8fafc;color:#475569;white-space:nowrap}.cp{background:var(--ps);border-color:#d7d7ff;color:var(--pd)}.ct{background:var(--ts);border-color:#a9eee4;color:#04786e}.cw{background:var(--ws);border-color:#fed7aa;color:#b45309}.cd{background:var(--ds);border-color:#fecdd3;color:#b91c1c}.cm{background:#f1f5f9;color:#64748b}.sg{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px}.tile{background:#eef3ff;border:1px solid #dbe4fb;border-radius:10px;padding:15px}.tile .k{font-size:calc(12px * var(--fs-scale));color:#475569;font-weight:750}.tile .v{font-family:"Microsoft YaHei UI","Microsoft YaHei","微软雅黑",sans-serif;font-size:calc(26px * var(--fs-scale));font-weight:900;color:var(--pd);margin-top:6px}.summary-overview .tile .k{font-size:calc(12px * var(--fs-scale-2x))!important;font-weight:950;color:#1D2939}.notice,.warning,.danger{border-radius:12px;padding:12px 14px;font-size:calc(13px * var(--fs-scale));line-height:1.65}.notice{border:1px solid #bfe8f3;background:#effbff;color:#164e63}.warning{border:1px solid #fed7aa;background:#fff7ed;color:#9a3412}.danger{border:1px solid #fecdd3;background:#fff1f2;color:#991b1b}.visits{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:18px}.visit{border:1px solid var(--line);border-radius:14px;background:#fff;padding:16px}.visit.latest{background:#f0f7ff;border-color:#bcd7ff}.vt{display:flex;justify-content:space-between;margin-bottom:10px;font-weight:850}.vb{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.vk{font-size:calc(12px * var(--fs-scale));color:var(--muted)}.vv{font-family:"Microsoft YaHei UI","Microsoft YaHei","微软雅黑",sans-serif;font-weight:850}.mw{overflow-x:auto;border:1px solid var(--line);border-radius:8px;background:#fff}.matrix{min-width:980px;border-collapse:separate;border-spacing:0;width:100%;font-family:"Microsoft YaHei UI","Microsoft YaHei","微软雅黑",sans-serif}.matrix th,.matrix td{border-right:1px solid var(--line);border-bottom:0;padding:17px 20px;text-align:center}.matrix th{background:#eef3ff;color:#334155;font-size:calc(13px * var(--fs-scale));font-weight:850}.matrix td{font-family:"Microsoft YaHei UI","Microsoft YaHei","微软雅黑",sans-serif;font-weight:800}.matrix th:first-child,.matrix td:first-child{position:sticky;left:0;text-align:left;background:#eef3ff;font-family:inherit;z-index:1;min-width:170px}.matrix th:last-child,.matrix td:last-child{border-right:0}.row-label{font-weight:900}.row-fsh{color:#d94663}.row-lh{color:#f59e0b}.row-hmg{color:#2563eb}.row-e2{color:#0f766e}.row-lhv{color:#7c3aed}.row-p{color:#475569}.row-oocyte{color:#4f46e5}.row-mii{color:#4f46e5}.row-ohss{color:#14b8a6}.today{background:#fff}.pred{background:#f0f5ff;color:#243b75}.pred-note{display:block;margin-top:3px;font-size:calc(11px * var(--fs-scale));color:#64748b;font-weight:700}.future{background:#f6f7fb;color:#94a3b8}.tbl{width:100%;border-collapse:collapse;background:#fff;border:1px solid var(--line);border-radius:16px;overflow:hidden}.tbl th{background:#eef3ff;color:#475569;text-align:left;font-size:calc(12px * var(--fs-scale));padding:13px 12px;border-bottom:1px solid var(--line)}.tbl td{padding:14px 12px;border-bottom:1px solid #edf1f7;font-size:calc(13px * var(--fs-scale))}.mono{font-family:"Microsoft YaHei UI","Microsoft YaHei","微软雅黑",sans-serif}.rank{width:28px;height:28px;border-radius:999px;background:var(--p);color:#fff;display:grid;place-items:center;font-weight:900}.alink{display:inline-flex;border:1px solid #d7d7ff;border-radius:8px;color:var(--pd);padding:7px 10px;font-weight:850;background:#fff}.status{position:fixed;left:0;right:0;bottom:0;height:38px;background:#eef3ff;border-top:1px solid var(--line);z-index:20;display:flex;align-items:center;justify-content:space-between;padding:0 28px;color:#334155;font-size:calc(12px * var(--fs-scale));font-weight:750}.links{display:flex;gap:24px}.quick{display:grid;grid-template-columns:1.45fr 1fr;gap:20px}.plist{width:100%;border-collapse:collapse}.plist th{font-size:calc(12px * var(--fs-scale));color:#64748b;background:#eef3ff;text-align:left;padding:11px}.plist td{padding:13px 11px;border-bottom:1px solid #edf1f7;font-size:calc(13px * var(--fs-scale))}.qcard{border:1px solid var(--line);border-radius:14px;background:#fff;padding:18px;display:flex;justify-content:space-between;gap:14px;align-items:center;margin-bottom:12px}.curves,.cases,.shap-top,.shap-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}.curve-sensitivity{border:1px solid #e2e8f0;border-radius:12px;background:#fbfdff;margin:10px 0 12px;overflow:hidden}.curve-s-head,.curve-s-row{display:grid;grid-template-columns:86px repeat(3,minmax(0,1fr));gap:8px;align-items:center}.curve-s-head{background:#eef3ff;color:#475569;font-size:calc(11px * var(--fs-scale));font-weight:900;padding:8px 10px}.curve-s-row{padding:9px 10px;border-top:1px solid #edf1f7}.curve-s-label{font-size:calc(12px * var(--fs-scale));font-weight:900;color:#0b1c30}.curve-s-cell{min-width:0}.curve-s-val{font-size:calc(13px * var(--fs-scale));font-weight:950;color:#0b1c30}.curve-s-dose{display:block;font-size:calc(10px * var(--fs-scale));color:#64748b;font-weight:800;margin-bottom:2px}.curve-s-delta{display:inline-flex;margin-top:3px;border-radius:999px;padding:2px 7px;font-size:calc(11px * var(--fs-scale));font-weight:900;border:1px solid #dbe4fb;background:#fff;color:#64748b}.curve-s-delta.up{border-color:#a9eee4;background:#dcfbf6;color:#04786e}.curve-s-delta.down{border-color:#fecdd3;background:#fff1f2;color:#b91c1c}.curve-s-delta.warn{border-color:#fed7aa;background:#fff7ed;color:#b45309}.curve-s-delta.flat{border-color:#e2e8f0;background:#f8fafc;color:#64748b}.tech-sens{margin-top:18px}.tech-dose-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin-top:14px}.tech-dose-card{border:1px solid #dbe4fb;border-radius:14px;background:#fff;padding:14px;min-width:0}.tech-dose-head{display:flex;justify-content:space-between;align-items:flex-start;gap:10px;margin-bottom:12px}.tech-dose-title{font-size:calc(15px * var(--fs-scale));font-weight:950;color:#0b1c30}.tech-dose-sub{font-size:calc(11px * var(--fs-scale));color:#64748b;margin-top:3px;line-height:1.35}.tech-dose-row{border-top:1px solid #edf1f7;padding:11px 0}.tech-dose-row:first-of-type{border-top:0}.tech-dose-row-title{font-size:calc(12px * var(--fs-scale));font-weight:900;color:#334155;margin-bottom:8px}.tech-dose-metrics{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}.tech-dose-metric{min-width:0}.tech-dose-top{display:flex;justify-content:space-between;gap:6px;align-items:center;font-size:calc(11px * var(--fs-scale));color:#64748b;font-weight:850}.tech-dose-top b{font-size:calc(11px * var(--fs-scale));color:#0b1c30;white-space:nowrap}.tech-dose-track{height:8px;border-radius:999px;background:#eef2f7;overflow:hidden;margin:5px 0 4px}.tech-dose-fill{height:100%;border-radius:999px;background:#94a3b8}.tech-dose-fill.up{background:var(--t)}.tech-dose-fill.down{background:#ef5f73}.tech-dose-fill.warn{background:var(--w)}.tech-dose-fill.flat{background:#cbd5e1}.tech-dose-value{font-size:calc(10px * var(--fs-scale));color:#64748b;font-weight:800;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}@media(max-width:1000px){.tech-dose-grid,.tech-dose-metrics{grid-template-columns:1fr}}.svgcard,.case,.shap-card,.dose-card{border:1px solid var(--line);border-radius:16px;background:#fff;padding:18px}.dose-card{display:flex;justify-content:space-between;align-items:center}.dose-card .value{font-size:calc(30px * var(--fs-scale));font-weight:900;color:var(--pd)}.factor{display:grid;grid-template-columns:132px 1fr;gap:14px;align-items:center;margin:14px 0}.bar{height:10px;background:#eef2f7;border-radius:999px;overflow:hidden}.fill{height:100%;border-radius:999px;background:var(--p)}.fill.t{background:var(--t)}.fill.w{background:var(--w)}.fc{grid-column:2;font-size:calc(12px * var(--fs-scale));color:#64748b;margin-top:-8px}.groups{display:grid;grid-template-columns:1fr 1.05fr;gap:26px;align-items:center}.grow{display:grid;grid-template-columns:178px 1fr;gap:14px;align-items:center;margin:14px 0}.gfill{height:100%;background:linear-gradient(90deg,var(--p),var(--t));border-radius:999px}.field-label{font-size:calc(14px * var(--fs-scale));color:#0f1f33;font-weight:850;margin:0 0 7px;min-height:42px;line-height:1.35}.field-label .req{color:#dc2626;font-weight:950}.monitor-record-title{font-size:calc(17px * var(--fs-scale));font-weight:950;color:#0b1c30;margin:2px 0 16px}.monitor-section-title{font-size:calc(16px * var(--fs-scale));font-weight:950;color:#172554;margin:22px 0 12px;padding:8px 0 8px 12px;border-left:4px solid var(--p);background:#f8fbff;border-radius:8px}.mini-hint{margin:8px 0 4px;padding:9px 11px;border-radius:10px;border:1px solid #fed7aa;background:#fff7ed;color:#9a3412;font-size:calc(12px * var(--fs-scale));font-weight:750}.monitor-empty{border:1.5px dashed #b7c3d8;border-radius:14px;background:#fbfdff;padding:18px;color:#64748b;margin-bottom:16px}.stTextInput input,.stNumberInput input,[data-baseweb="input"] input{background:transparent!important;color:var(--ink)!important;-webkit-text-fill-color:var(--ink)!important;font-weight:720!important}.stTextInput>div>div,.stNumberInput>div>div,[data-baseweb="input"]>div,[data-baseweb="select"]>div{background:#fff!important;border:1px solid #94A3B8!important;border-radius:8px!important;min-height:44px!important;box-shadow:none!important}.stTextInput>div>div:focus-within,.stNumberInput>div>div:focus-within,[data-baseweb="input"]>div:focus-within,[data-baseweb="select"]>div:focus-within{border-color:#475569!important;box-shadow:0 0 0 1px #94A3B8!important}.stTextInput label p,.stNumberInput label p,.stSelectbox label p,.stDateInput label p{font-weight:800!important;color:#334155!important}div[data-testid="stButton"]>button,div[data-testid="stFormSubmitButton"] button{border-radius:9px!important;border:1px solid #c7c4d8!important;background:#fff!important;color:var(--pd)!important;min-height:42px!important;font-weight:850!important;box-shadow:none!important}button[kind="primary"],button[data-testid*="primary"],div[data-testid="stButton"]>button[kind="primary"],div[data-testid="stFormSubmitButton"] button[kind="primary"]{background:var(--p)!important;border-color:var(--p)!important;color:#fff!important}@media(max-width:1000px){div[role="radiogroup"]{justify-content:flex-start;overflow-x:auto;margin-top:0;margin-bottom:20px}.topbar{margin-bottom:12px}.grid4,.sg,.curves,.cases,.shap-top,.shap-grid,.quick,.visits,.groups{grid-template-columns:1fr}.block-container{padding-left:16px!important;padding-right:16px!important}.topbar{margin-left:-16px;margin-right:-16px;padding:0 16px}.status{position:static;margin:28px -16px -82px}}
div[data-testid="stFormSubmitButton"] button[kind*="primary"],div[data-testid="stFormSubmitButton"] button[data-testid*="primary"]{background:var(--p)!important;border-color:var(--p)!important;color:#fff!important}
.tbl th{border-bottom:0!important}
.dash-dose{height:44px;border:1.6px solid #b8c5d9;border-radius:8px;background:#f8fbff;display:flex;align-items:center;padding:0 11px;color:#64748b;font-weight:850}.dash-dose b{color:#0b1c30;margin-right:6px}
.model-strip{border:1px solid #dbe4fb;background:linear-gradient(180deg,#fff,#f8fbff);border-radius:14px;padding:12px 14px;margin:0 0 16px;display:flex;align-items:center;justify-content:space-between;gap:14px;box-shadow:0 4px 10px rgba(15,23,42,.035)}.model-strip-left{display:flex;align-items:center;gap:10px;min-width:0}.model-dot{width:10px;height:10px;border-radius:999px;background:var(--t);box-shadow:0 0 0 4px rgba(20,184,166,.12);display:inline-block;flex:0 0 auto}.model-dot.running{background:var(--p);box-shadow:0 0 0 4px rgba(79,70,229,.12)}.model-dot.warn{background:var(--w);box-shadow:0 0 0 4px rgba(245,158,11,.14)}.model-dot.err{background:var(--d);box-shadow:0 0 0 4px rgba(220,38,38,.12)}.model-strip-title{font-size:calc(13px * var(--fs-scale));font-weight:900;color:#0b1c30}.model-strip-sub{font-size:calc(12px * var(--fs-scale));color:#64748b;margin-top:2px;line-height:1.4}.model-strip-tags{display:flex;gap:8px;flex-wrap:wrap;justify-content:flex-end}.model-running{border:1px solid #d7d7ff;background:#f7f8ff;color:#3525cd;border-radius:12px;padding:11px 13px;margin:0 0 14px;font-size:calc(13px * var(--fs-scale));font-weight:850;display:flex;gap:10px;align-items:center}@media(max-width:1000px){.model-strip{align-items:flex-start;flex-direction:column}.model-strip-tags{justify-content:flex-start}}
.breakdown-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px;margin-top:18px}.bd-card{border:1px solid var(--line);border-radius:16px;background:#fff;padding:18px;min-width:0}.bd-head{display:flex;align-items:flex-start;justify-content:space-between;gap:12px;margin-bottom:14px}.bd-title{font-weight:900;color:#0b1c30;font-size:calc(15px * var(--fs-scale));line-height:1.45}.bd-target{font-size:calc(12px * var(--fs-scale));color:#64748b;margin-top:3px}.bd-stack{display:flex;flex-direction:column;gap:14px}.bd-row{display:grid;grid-template-columns:minmax(118px,150px) minmax(0,1fr);column-gap:14px;row-gap:6px;align-items:center}.bd-lab{font-weight:850;color:#0b1c30;font-size:calc(13px * var(--fs-scale));line-height:1.35}.bd-val{font-size:calc(12px * var(--fs-scale));color:#64748b;margin-top:2px}.bd-track{height:10px;background:#eef2f7;border-radius:999px;overflow:hidden}.bd-fill{height:100%;border-radius:999px;background:var(--t)}.bd-fill.neg{background:#ef5f73}.bd-meta{grid-column:2;display:flex;justify-content:flex-start;align-items:center;gap:8px;margin-top:-2px}.bd-chip{display:inline-flex;border-radius:999px;padding:4px 10px;font-size:calc(12px * var(--fs-scale));font-weight:850;border:1px solid #a9eee4;background:var(--ts);color:#04786e}.bd-chip.neg{border-color:#fecdd3;background:var(--ds);color:#b91c1c}.bd-score{font-size:calc(12px * var(--fs-scale));color:#64748b;font-weight:750}.bd-note{font-size:calc(12px * var(--fs-scale));color:#64748b;line-height:1.55;margin-top:14px}.tech-shap{margin-top:18px;border:1px solid #dbe4fb;border-radius:14px;background:#fbfdff;overflow:hidden}.tech-shap summary{cursor:pointer;list-style:none;display:flex;align-items:center;justify-content:space-between;gap:12px;padding:14px 16px;font-weight:900;color:#0b1c30}.tech-shap summary::-webkit-details-marker{display:none}.tech-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;padding:0 16px 16px}.tech-card{border:1px solid var(--line);border-radius:12px;background:#fff;padding:14px;min-width:0}.tech-title{font-size:calc(13px * var(--fs-scale));font-weight:900;color:#334155;margin-bottom:10px}.tech-row{display:grid;grid-template-columns:minmax(112px,1fr) 84px 58px;gap:10px;align-items:center;padding:8px 0;border-top:1px solid #edf1f7}.tech-row:first-of-type{border-top:0}.tech-label{font-size:calc(12px * var(--fs-scale));font-weight:850;color:#0b1c30;line-height:1.35}.tech-value{font-size:calc(11px * var(--fs-scale));color:#64748b;margin-top:2px}.tech-score{font-size:calc(12px * var(--fs-scale));font-weight:850;color:#64748b;text-align:right}.tech-score.neg{color:#b91c1c}
.nav-i{position:relative;color:#0b1c30;border:1px solid transparent;transition:background .16s ease,color .16s ease,border-color .16s ease}.nav-i svg{width:16px;height:16px;display:block;stroke:currentColor;fill:none;stroke-width:2;stroke-linecap:round;stroke-linejoin:round}.nav-i:hover{background:var(--ps);color:var(--pd);border-color:#d7d7ff}.nav-i.notice:after{content:"";position:absolute;right:8px;top:8px;width:6px;height:6px;border-radius:999px;background:var(--t);border:1px solid #fff}
div[data-testid="stForm"]{background:#fff!important;border:1px solid var(--line)!important;border-radius:16px!important;padding:22px 16px 14px!important;box-shadow:0 4px 12px rgba(15,23,42,.04)!important}.tile.dose .v{font-size:calc(26px * var(--fs-scale))}.tile .dose-code{display:inline-flex;margin-right:6px;border-radius:999px;padding:2px 8px;background:#fff;border:1px solid #d7d7ff;color:var(--pd);font-family:"JetBrains Mono",monospace;font-size:calc(11px * var(--fs-scale));font-weight:850}
.curves,.cases,.shap-top,.shap-grid,.breakdown-grid{grid-template-columns:repeat(3,minmax(0,1fr))}.svgcard,.case,.shap-card,.dose-card,.bd-card{min-width:0}.shap-card .note,.case .note{overflow-wrap:anywhere;word-break:break-word}.factor{grid-template-columns:minmax(112px,132px) minmax(0,1fr)}
@media(max-width:1000px){.breakdown-grid,.tech-grid{grid-template-columns:1fr!important}}
.breakdown-grid.gn-explain-grid,.breakdown-grid.outcome-explain-grid{grid-template-columns:repeat(3,minmax(0,1fr))}.breakdown-row-title{font-size:calc(13px * var(--fs-scale));font-weight:900;color:#475569;margin:18px 0 -2px}.ohss-grid{display:grid;grid-template-columns:1.1fr .95fr;gap:18px;align-items:stretch}.ohss-card{border:1px solid var(--line);border-radius:16px;background:#fff;padding:20px;min-width:0}.ohss-card.primary{background:#f8fbff;border-color:#cfdcf6}.ohss-title{display:flex;align-items:flex-start;justify-content:space-between;gap:12px;margin-bottom:16px}.ohss-title h3{margin:0;font-size:calc(20px * var(--fs-scale));color:#0b1c30}.ohss-title .en{font-size:calc(12px * var(--fs-scale));color:#64748b;margin-top:4px;font-weight:750}.ohss-main{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}.ohss-k{font-size:calc(12px * var(--fs-scale));color:#64748b;font-weight:800}.ohss-v{font-size:calc(28px * var(--fs-scale));font-weight:950;color:var(--pd);margin-top:5px}.ohss-sub{font-size:calc(12px * var(--fs-scale));color:#475569;line-height:1.55;margin-top:14px}.ohss-factor-grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.ohss-factor-head{font-size:calc(14px * var(--fs-scale));font-weight:900;color:#0b1c30;margin-bottom:12px}.risk-row{display:grid;grid-template-columns:minmax(128px,166px) 1fr;gap:12px;align-items:center;margin:12px 0}.risk-name{font-size:calc(13px * var(--fs-scale));font-weight:850;color:#0b1c30;line-height:1.35}.risk-val{font-size:calc(12px * var(--fs-scale));color:#64748b;margin-top:2px}.risk-track{height:10px;border-radius:999px;background:#eef2f7;overflow:hidden}.risk-fill{height:100%;border-radius:999px;background:var(--d)}.risk-fill.down{background:var(--t)}.risk-meta{grid-column:2;display:flex;gap:8px;align-items:center;margin-top:-4px}.risk-note{font-size:calc(12px * var(--fs-scale));color:#64748b;line-height:1.55;margin-top:12px}.safety-concern{display:inline-flex;border-radius:999px;padding:4px 10px;font-size:calc(12px * var(--fs-scale));font-weight:850;border:1px solid #fed7aa;background:#fff7ed;color:#b45309}.safety-concern.high{border-color:#fecdd3;background:#fff1f2;color:#b91c1c}@media(max-width:1000px){.ohss-grid,.ohss-main,.ohss-factor-grid{grid-template-columns:1fr}}
.bd-prob-flow{display:flex;align-items:center;justify-content:flex-end;gap:8px;flex:0 0 auto}.bd-base{display:flex;align-items:baseline;gap:6px;color:#64748b;font-size:calc(11px * var(--fs-scale));font-weight:750;white-space:nowrap}.bd-base strong{color:#334155;font-size:calc(13px * var(--fs-scale));font-weight:900}.bd-arrow{color:#6366f1;font-size:calc(18px * var(--fs-scale));font-weight:900;line-height:1}.bd-row.other{padding-top:12px;border-top:1px solid #e5eaf2}.bd-row.other .bd-lab{color:#334155}.bd-row.other .bd-val{font-size:calc(11px * var(--fs-scale))}.bd-other-details{margin-top:2px;border-top:1px solid #dfe6ef}.bd-other-details summary{cursor:pointer;list-style:none;padding-top:13px;border-top:0}.bd-other-details summary::-webkit-details-marker{display:none}.bd-other-toggle{display:inline-flex;align-items:center;border:1px solid #cbd5e1;border-radius:6px;padding:3px 7px;background:#fff;color:#475569;font-size:calc(11px * var(--fs-scale));font-weight:850}.bd-other-toggle .opened{display:none}.bd-other-details[open] .bd-other-toggle .closed{display:none}.bd-other-details[open] .bd-other-toggle .opened{display:inline}.bd-other-list{margin-top:10px;padding:8px 10px;border:1px solid #e3eaf2;border-radius:8px;background:#f8fafc;max-height:250px;overflow:auto}.bd-other-item{display:grid;grid-template-columns:minmax(0,1fr) auto minmax(76px,auto);gap:10px;align-items:center;padding:8px 2px;border-top:1px solid #e8edf3}.bd-other-item:first-child{border-top:0}.bd-other-name{font-size:calc(12px * var(--fs-scale));font-weight:850;color:#27364a}.bd-other-value{font-size:calc(11px * var(--fs-scale));color:#64748b;margin-top:2px}.bd-other-item strong{font-size:calc(11px * var(--fs-scale));color:#475569;text-align:right;white-space:nowrap}.bd-mini-chip{display:inline-flex;border-radius:999px;padding:3px 7px;font-size:calc(10px * var(--fs-scale));font-weight:850;border:1px solid #a9eee4;background:#ecfdf9;color:#04786e;white-space:nowrap}.bd-mini-chip.down{border-color:#fecdd3;background:#fff1f2;color:#b91c1c}.bd-other-empty{padding:10px;color:#64748b;font-size:calc(12px * var(--fs-scale));text-align:center}@media(max-width:680px){.bd-head{align-items:flex-start}.bd-prob-flow{align-items:flex-end;flex-direction:column;gap:4px}.bd-arrow{display:none}.bd-other-item{grid-template-columns:minmax(0,1fr) auto}.bd-other-item strong{grid-column:2}.bd-other-toggle{padding:3px 6px}}
.breakdown-grid.gn-explain-grid,.breakdown-grid.outcome-explain-grid{gap:16px;align-items:stretch}.bd-card{display:flex;flex-direction:column;height:100%;padding:17px;border-radius:14px;background:#fff}.bd-card.probability-card{border-color:#d8dcfa}.bd-card.outcome-card{border-color:#cfe8e5;background:#fcfefe}.bd-head{min-height:54px;margin-bottom:16px;padding-bottom:13px;border-bottom:1px solid #edf1f7}.bd-title{font-size:calc(15px * var(--fs-scale));line-height:1.35}.bd-target{margin-top:4px;line-height:1.4}.bd-stack{flex:1;gap:13px}.bd-row{grid-template-columns:minmax(112px,142px) minmax(0,1fr);column-gap:13px;row-gap:5px;margin:0}.bd-track{height:8px}.bd-meta{gap:7px}.bd-score{font-weight:850;color:#475569}.bd-row.other{margin-top:2px;padding-top:13px;border-top:1px solid #dfe6ef}.bd-fill.neutral{background:#94a3b8}.bd-chip.neutral{border-color:#cbd5e1;background:#f8fafc;color:#64748b}.breakdown-row-title{display:flex;align-items:center;min-height:28px;margin:22px 0 11px;padding-left:10px;border-left:3px solid var(--p);font-size:calc(14px * var(--fs-scale));font-weight:900;color:#27364a}.breakdown-row-title.outcome-title{border-left-color:var(--t)}@media(max-width:1000px){.breakdown-grid.gn-explain-grid,.breakdown-grid.outcome-explain-grid{grid-template-columns:1fr}.bd-card{height:auto}}@media(max-width:520px){.bd-row{grid-template-columns:minmax(100px,128px) minmax(0,1fr)}.bd-card{padding:15px}}
[data-testid="InputInstructions"],[data-testid="stInputInstructions"]{display:none!important}
.stTextInput [data-baseweb="input"],.stNumberInput [data-baseweb="input"],.stDateInput [data-baseweb="input"],[data-baseweb="select"]>div{background:#fff!important;border:1.5px solid #94a3b8!important;border-radius:8px!important;min-height:44px!important;box-shadow:inset 0 0 0 .5px rgba(148,163,184,.32)!important;overflow:hidden!important}
.stTextInput [data-baseweb="input"]>div,.stNumberInput [data-baseweb="input"]>div,.stDateInput [data-baseweb="input"]>div{border:0!important;box-shadow:none!important}
.stTextInput [data-baseweb="input"]:focus-within,.stNumberInput [data-baseweb="input"]:focus-within,.stDateInput [data-baseweb="input"]:focus-within,[data-baseweb="select"]>div:focus-within{border-color:#475569!important;box-shadow:0 0 0 1px #94a3b8!important}
.response-line{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-top:6px}.response-line .v{margin-top:0}.response-strategy{display:flex;align-items:center;justify-content:space-between;gap:14px;border:1px solid #cfe8e5;background:#f7fffd;border-radius:12px;padding:11px 14px;margin:0 0 16px}.response-strategy-main{display:flex;align-items:center;gap:9px;flex-wrap:wrap;font-size:calc(13px * var(--fs-scale));color:#334155}.response-strategy-main b{color:#0b1c30}.response-strategy-rule{font-size:calc(12px * var(--fs-scale));color:#64748b;font-weight:750}.response-low{background:#eef0ff;border-color:#d7d7ff;color:#3525cd}.response-normal{background:#dcfbf6;border-color:#a9eee4;color:#04786e}.response-high{background:#fff7ed;border-color:#fed7aa;color:#b45309}
@media(max-width:700px){.response-strategy{align-items:flex-start;flex-direction:column}.response-strategy-rule{line-height:1.55}}
.knn-evidence-head{display:flex;align-items:center;gap:10px;margin:26px 0 14px}.knn-evidence-head .ic{width:28px;height:28px;border-radius:8px;background:var(--ps);color:var(--pd);display:grid;place-items:center;font-weight:900}.knn-evidence-head h3{margin:0;font-size:calc(20px * var(--fs-scale))}.knn-drug-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}.knn-drug-card{border:1px solid var(--line);border-radius:12px;background:#fff;overflow:hidden;min-width:0}.knn-drug-title{display:flex;align-items:center;justify-content:space-between;gap:10px;padding:14px 15px;border-bottom:1px solid #e5eaf2;color:#0b1c30;font-size:calc(15px * var(--fs-scale));font-weight:900}.knn-mini-table{width:100%;border-collapse:collapse;table-layout:fixed}.knn-mini-table th{background:#eef3ff;color:#475569;font-size:calc(11px * var(--fs-scale));font-weight:850;padding:10px 7px;text-align:center;border:0}.knn-mini-table td{font-size:calc(12px * var(--fs-scale));font-weight:750;padding:11px 7px;text-align:center;border-top:1px solid #edf1f7;white-space:nowrap}.knn-mini-table td:first-child{font-weight:900;color:var(--pd)}.knn-evidence-note{font-size:calc(12px * var(--fs-scale));color:#64748b;line-height:1.55;margin-top:10px}@media(max-width:1000px){.knn-drug-grid{grid-template-columns:1fr}}
.curve-rec-summary{display:grid;grid-template-columns:1.15fr .8fr .9fr;gap:8px;margin:13px 0 2px;padding:8px 10px;border:1px solid #dbe4fb;border-radius:8px;background:#f8fbff;color:#64748b;font-size:calc(11px * var(--fs-scale));font-weight:750}.curve-rec-summary span{min-width:0;white-space:nowrap}.curve-rec-summary b{color:#27364a;font-weight:900}.curve-rec-summary .dose{color:#3525cd;font-weight:900}
.stNumberInput button{border-left:1px solid #cbd5e1!important}
.stApp [data-testid="stVerticalBlockBorderWrapper"]{background:#fff!important;border:1px solid var(--line)!important;border-radius:14px!important;padding:16px 18px 6px!important;box-shadow:0 2px 8px rgba(15,23,42,.03)!important;margin-bottom:18px}
.stApp [data-testid="stVerticalBlockBorderWrapper"]:focus-within{border-color:#c7c4d8!important}
.stTextInput:hover [data-baseweb="input"],.stNumberInput:hover [data-baseweb="input"],.stDateInput:hover [data-baseweb="input"],[data-baseweb="select"]>div:hover{border-color:#818cf8!important}
.stApp .stSelectbox [data-baseweb="select"]>div,.stApp .stDateInput [data-baseweb="input"]{min-height:44px!important;box-sizing:border-box!important}
.stApp .stSelectbox [data-baseweb="select"]>div,[data-baseweb="select"]>div,div[data-testid="stFormSubmitButton"] button{border-radius:8px!important}
div[data-testid="stFormSubmitButton"] button:hover{border-color:#818cf8!important;background:#f5f6ff!important}
/* 推荐解释页图表（SHAP 条形图）稍降一档避免长特征名在标签列换行 */
.bd-card,.tech-shap{--fs-scale:var(--fs-scale-tight)}
</style>
"""

# 患者临床信息录入页专属覆盖样式：医疗 B 端专业干净风格。
# 仅通过 patient_page() 注入，切到其他页面时自动移除，不影响监测结果/决策曲线/推荐解释。
PATIENT_CSS = """
<style>
/* 页面背景 */
html,body,.stApp,[data-testid="stAppViewContainer"],[data-testid="stAppViewBlockContainer"]{background:#F9FAFB!important}

/* 页面标题 20px/600/#1D2939；副标题辅助说明 */
.title{margin:0 0 20px!important}
.title h1{font-size:36px!important;font-weight:600!important;color:#1D2939!important;margin:0 0 6px!important;line-height:1.4}
.title p{font-size:13px!important;font-weight:400!important;color:#667085!important;line-height:1.6;margin:0!important}

/* 一级模块标题 18px/600/#344054（患者基础信息、监测记录） */
.head{margin-bottom:14px!important}
.head .ic{width:24px!important;height:24px!important;border-radius:6px!important;background:#EFF4FF!important;color:#3B82F6!important;font-size:calc(13px * var(--fs-scale))!important}
.head h3{font-size:calc(18px * var(--fs-scale))!important;font-weight:600!important;color:#344054!important}

/* 主表单卡片：白底/10px圆角/1px #E5E7EB/内边距16px */
div[data-testid="stForm"]{background:#fff!important;border:1px solid #E5E7EB!important;border-radius:10px!important;padding:16px!important;box-shadow:none!important}

/* 单条监测记录 = 嵌套子卡片：12px圆角白卡，间距12px */
.stApp [data-testid="stVerticalBlockBorderWrapper"]{background:#fff!important;border:1px solid #E5E7EB!important;border-radius:10px!important;padding:14px 16px!important;box-shadow:none!important;margin-bottom:12px}
.stApp [data-testid="stVerticalBlockBorderWrapper"]:focus-within{border-color:#D0D5DD!important}

/* “第 N 次监测记录”标题 */
.monitor-record-title{font-size:calc(14px * var(--fs-scale))!important;font-weight:600!important;color:#344054!important;margin:0 0 12px!important}

/* 二级分组标题 15px/500/#475467 + 极浅 #F2F4F7 细分隔线 */
.monitor-section-title{font-size:calc(15px * var(--fs-scale))!important;font-weight:500!important;color:#475467!important;margin:14px 0 10px!important;padding:12px 0 0!important;border-top:1px solid #F2F4F7!important;border-left:0!important;background:transparent!important;border-radius:0!important}

/* 表单 label 13px/400/#475467；固定两行高保证同排输入框对齐 */
.field-label{font-size:calc(13px * var(--fs-scale))!important;font-weight:400!important;color:#475467!important;margin:0 0 6px!important;min-height:40px;line-height:1.4}
.field-label .req{color:#F04438!important;font-weight:500!important}

/* 输入框/下拉/日期：统一36px高、6px圆角、默认#D0D5DD、聚焦#3B82F6 */
.stApp .stTextInput [data-baseweb="input"],
.stApp .stNumberInput [data-baseweb="input"],
.stApp .stDateInput [data-baseweb="input"],
.stApp [data-baseweb="select"]>div{
  min-height:36px!important;height:36px!important;
  border-radius:6px!important;border:1px solid #D0D5DD!important;
  background:#fff!important;box-shadow:none!important;
  font-size:calc(14px * var(--fs-scale))!important
}
.stApp .stTextInput [data-baseweb="input"]:focus-within,
.stApp .stNumberInput [data-baseweb="input"]:focus-within,
.stApp .stDateInput [data-baseweb="input"]:focus-within,
.stApp [data-baseweb="select"]>div:focus-within{
  border-color:#3B82F6!important;box-shadow:0 0 0 1px #3B82F6!important;outline:none!important
}
/* 清除 BaseWeb 在 [data-baseweb="input"] 上挂的伪元素聚焦光晕（红色） */
.stApp .stTextInput [data-baseweb="input"]::before,
.stApp .stNumberInput [data-baseweb="input"]::before,
.stApp .stDateInput [data-baseweb="input"]::before,
.stApp [data-baseweb="select"]>div::before,
.stApp .stTextInput [data-baseweb="input"]::after,
.stApp .stNumberInput [data-baseweb="input"]::after,
.stApp .stDateInput [data-baseweb="input"]::after,
.stApp [data-baseweb="select"]>div::after{
  border:0!important;border-color:transparent!important;background:transparent!important;box-shadow:none!important;outline:none!important
}
/* 清除 BaseWeb 新版 [data-baseweb="base-input"] 内层聚焦光晕（这是真正的红色来源） */
.stApp .stTextInput [data-baseweb="base-input"],
.stApp .stNumberInput [data-baseweb="base-input"],
.stApp .stDateInput [data-baseweb="base-input"],
.stApp [data-baseweb="select"]>div>div [data-baseweb="base-input"],
.stApp .stTextInput [data-baseweb="base-input"]:focus,
.stApp .stNumberInput [data-baseweb="base-input"]:focus,
.stApp .stDateInput [data-baseweb="base-input"]:focus,
.stApp [data-baseweb="select"]>div>div [data-baseweb="base-input"]:focus,
.stApp .stTextInput [data-baseweb="base-input"]:focus-within,
.stApp .stNumberInput [data-baseweb="base-input"]:focus-within,
.stApp .stDateInput [data-baseweb="base-input"]:focus-within,
.stApp [data-baseweb="select"]>div>div [data-baseweb="base-input"]:focus-within{
  border:0!important;border-color:transparent!important;background:transparent!important;box-shadow:none!important;outline:none!important
}
.stApp .stTextInput [data-baseweb="base-input"]::before,
.stApp .stNumberInput [data-baseweb="base-input"]::before,
.stApp .stDateInput [data-baseweb="base-input"]::before,
.stApp [data-baseweb="select"]>div>div [data-baseweb="base-input"]::before,
.stApp .stTextInput [data-baseweb="base-input"]::after,
.stApp .stNumberInput [data-baseweb="base-input"]::after,
.stApp .stDateInput [data-baseweb="base-input"]::after,
.stApp [data-baseweb="select"]>div>div [data-baseweb="base-input"]::after{
  border:0!important;border-color:transparent!important;background:transparent!important;box-shadow:none!important;outline:none!important
}
.stApp .stTextInput [data-baseweb="input"]>div,
.stApp .stNumberInput [data-baseweb="input"]>div,
.stApp .stDateInput [data-baseweb="input"]>div{border:0!important;box-shadow:none!important}
.stApp .stTextInput input,.stApp .stNumberInput input,.stApp .stDateInput input,
.stApp [data-baseweb="select"]>div>div{
  font-size:calc(14px * var(--fs-scale))!important;font-weight:400!important;color:#1D2939!important;
  -webkit-text-fill-color:#1D2939!important;
  outline:none!important;box-shadow:none!important;border-color:transparent!important
}
.stApp [data-baseweb="select"]{color:#1D2939!important;font-size:calc(14px * var(--fs-scale))!important}

/* ±微调按钮缩小适配36px输入框 */
.stApp .stNumberInput button{width:22px!important;min-width:22px!important;padding:0!important;border-left:1px solid #E5E7EB!important}
.stApp .stNumberInput button svg{width:11px;height:11px}

/* 底部按钮：38px高、8px圆角、主蓝#3B82F6、次按钮白底灰边灰字 */
div[data-testid="stFormSubmitButton"] button{
  height:38px!important;min-height:38px!important;border-radius:8px!important;
  font-size:14px!important;font-weight:500!important;
  border:1px solid #D0D5DD!important;background:#fff!important;color:#344054!important;box-shadow:none!important
}
div[data-testid="stFormSubmitButton"] button[kind="primary"]{background:#3B82F6!important;border-color:#3B82F6!important;color:#fff!important}
div[data-testid="stFormSubmitButton"] button[kind="primary"]:hover{background:#2F71E6!important;border-color:#2F71E6!important}
div[data-testid="stFormSubmitButton"] button:hover{border-color:#98A2B3!important;background:#F9FAFB!important}

/* 列间距统一12px（含底部按钮间距） */
.stApp div[data-testid="stHorizontalBlock"]{gap:12px!important}

/* 辅助提示 12px/#F97316 */
.mini-hint{font-size:calc(12px * var(--fs-scale))!important;color:#F97316!important;background:#FFF7ED!important;border:1px solid #FED7AA!important;border-radius:6px!important;font-weight:400!important;padding:8px 10px!important;line-height:1.5}

/* 浏览器原生 <input> 上的 aria-invalid 红 outline 兜底 */
.stApp .stTextInput input[aria-invalid="true"],
.stApp .stNumberInput input[aria-invalid="true"],
.stApp .stDateInput input[aria-invalid="true"]{
  outline:none!important;box-shadow:none!important;border-color:transparent!important;
}
/* 精准修复：外层容器 stNumberInputContainer/stTextInputContainer/stDateInputContainer
   Streamlit 在容器聚焦时给 data-testid="stNumberInputContainer" 等加上 .focused 类，
   emotion CSS 把它画成红色 (rgb(255,75,75))。这里用 [data-testid] 而非 emotion 类名，
   避免 emotion cache hash 变化导致失效。
   关键：内层 [data-baseweb="input"] 在所有状态下都透明边框+无阴影，只让外层显示一条边框
   （默认灰、聚焦蓝），彻底消除双层边框。 */
.stApp [data-testid="stNumberInputContainer"],
.stApp [data-testid="stTextInputContainer"],
.stApp [data-testid="stDateInputContainer"]{
  border-color:#D0D5DD!important;
}
.stApp [data-testid="stNumberInputContainer"].focused,
.stApp [data-testid="stNumberInputContainer"]:focus-within,
.stApp [data-testid="stTextInputContainer"].focused,
.stApp [data-testid="stTextInputContainer"]:focus-within,
.stApp [data-testid="stDateInputContainer"].focused,
.stApp [data-testid="stDateInputContainer"]:focus-within{
  border-color:#3B82F6!important;
}
/* 内层 [data-baseweb="input"] 永远透明，无论是否聚焦。
   用 :is() 合并三类输入组件与对应容器 testid，特异性保持 (0,4,2)~(0,4,4)，
   击败旧 :focus-within 规则 (0,2,2)，避免聚焦时被染回蓝。 */
.stApp :is(.stNumberInput,.stTextInput,.stDateInput) :is([data-testid="stNumberInputContainer"],[data-testid="stTextInputContainer"],[data-testid="stDateInputContainer"]) [data-baseweb="input"],
.stApp :is(.stNumberInput,.stTextInput,.stDateInput) :is([data-testid="stNumberInputContainer"],[data-testid="stTextInputContainer"],[data-testid="stDateInputContainer"]):focus-within [data-baseweb="input"],
.stApp :is(.stNumberInput,.stTextInput,.stDateInput) :is([data-testid="stNumberInputContainer"],[data-testid="stTextInputContainer"],[data-testid="stDateInputContainer"]) [data-baseweb="input"]:focus-within,
.stApp :is(.stNumberInput,.stTextInput,.stDateInput) :is([data-testid="stNumberInputContainer"],[data-testid="stTextInputContainer"],[data-testid="stDateInputContainer"]):focus-within [data-baseweb="input"]:focus-within{
  border-color:transparent!important;
  box-shadow:none!important;
}
</style>
"""

BELL_ICON = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M18 8a6 6 0 0 0-12 0c0 7-3 9-3 9h18s-3-2-3-9"></path><path d="M13.7 21a2 2 0 0 1-3.4 0"></path></svg>'
SETTINGS_ICON = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 15.5a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7z"></path><path d="M19.4 15a1.8 1.8 0 0 0 .4 2l.1.1a2.1 2.1 0 0 1-3 3l-.1-.1a1.8 1.8 0 0 0-2-.4 1.8 1.8 0 0 0-1.1 1.7V21a2.1 2.1 0 0 1-4.2 0v-.2a1.8 1.8 0 0 0-1.2-1.7 1.8 1.8 0 0 0-2 .4l-.1.1a2.1 2.1 0 0 1-3-3l.1-.1a1.8 1.8 0 0 0 .4-2 1.8 1.8 0 0 0-1.7-1.1H2a2.1 2.1 0 0 1 0-4.2h.2a1.8 1.8 0 0 0 1.7-1.2 1.8 1.8 0 0 0-.4-2l-.1-.1a2.1 2.1 0 0 1 3-3l.1.1a1.8 1.8 0 0 0 2 .4 1.8 1.8 0 0 0 1.1-1.7V2a2.1 2.1 0 0 1 4.2 0v.2a1.8 1.8 0 0 0 1.2 1.7 1.8 1.8 0 0 0 2-.4l.1-.1a2.1 2.1 0 0 1 3 3l-.1.1a1.8 1.8 0 0 0-.4 2 1.8 1.8 0 0 0 1.7 1.1h.2a2.1 2.1 0 0 1 0 4.2h-.2a1.8 1.8 0 0 0-1.9 1.2z"></path></svg>'

def rerun(): st.rerun() if hasattr(st, "rerun") else st.experimental_rerun()
def clamp(x,a,b): return max(a,min(b,x))
def normalize_page(raw):
    if isinstance(raw, (list, tuple)):
        raw = raw[0] if raw else None
    if raw is None:
        return None
    value = str(raw).strip()
    page = LEGACY.get(value, SLUG_PAGES.get(value, value))
    return page if page in PAGES else None
def get_query_page():
    try:
        raw = st.query_params.get("view", None)
    except Exception:
        try:
            raw = st.experimental_get_query_params().get("view", None)
        except Exception:
            raw = None
    return normalize_page(raw)
def set_query_page(p):
    slug = PAGE_SLUGS.get(p)
    if not slug:
        return
    try:
        if st.query_params.get("view", None) != slug:
            st.query_params["view"] = slug
    except Exception:
        try:
            st.experimental_set_query_params(view=slug)
        except Exception:
            pass
def set_page(p):
    p = normalize_page(p) or "患者录入"
    st.session_state.page=p
    st.session_state._pending_page=p
    st.session_state._last_query_page=p
    set_query_page(p); rerun()
def on_nav_change():
    p = normalize_page(st.session_state.get("page_selector")) or st.session_state.get("page") or "患者录入"
    st.session_state.page = p
    st.session_state._last_query_page = p
    set_query_page(p)
def init():
    pending_page=normalize_page(st.session_state.pop("_pending_page", None))
    query_page=get_query_page()
    selector_page=normalize_page(st.session_state.get("page_selector"))
    stored_page=normalize_page(st.session_state.get("page"))
    last_query_page=normalize_page(st.session_state.get("_last_query_page"))
    if pending_page is not None:
        p=pending_page
    elif query_page is not None and (selector_page is None or query_page != last_query_page):
        p=query_page
    else:
        p=selector_page or stored_page or query_page or "患者录入"
    st.session_state.page=p
    st.session_state.page_selector=p
    if query_page is not None:
        st.session_state._last_query_page=query_page
    st.session_state.setdefault("patient", DEFAULT.copy()); st.session_state.setdefault("validation_error", ""); st.session_state.setdefault("doctor_final", {})
    st.session_state.setdefault("monitoring_records", default_monitoring_records())
    # Keep the data-entry page responsive; recommendation pages compute lazily via sync_recommendations().
    st.session_state.setdefault("recs", [])
    st.session_state.setdefault("recs_stale", False)
def header():
    st.markdown(CSS, unsafe_allow_html=True)
    # 顶部导航栏：单个容器内「品牌（绝对定位靠左）+ 页面标签 radio（整行居中）」，sticky 置顶
    nav = st.container()
    with nav:
        st.markdown(f'<div class="nav-brand"><div class="mark">⚕</div><div><div class="brand-name">Gn 剂量辅助系统</div><div class="sub">医生端临床辅助决策界面</div></div></div>', unsafe_allow_html=True)
        selected = st.radio("页面导航", PAGES, horizontal=True, label_visibility="collapsed", key="page_selector", on_change=on_nav_change)
    selected = normalize_page(selected) or st.session_state.get("page", "患者录入")
    st.session_state.page = selected
    if get_query_page() != selected:
        st.session_state._last_query_page = selected
        set_query_page(selected)
def statusbar(): st.markdown('<div class="status"><div>Gn 剂量辅助系统 v2.4.1 | 系统状态: 运行正常 (Lab Sync Active)</div><div class="links"><span>隐私政策</span><span>技术支持</span><span>操作手册</span></div></div>', unsafe_allow_html=True)
def title(h,s): st.markdown(f'<div class="title"><h1>{escape(h)}</h1><p>{escape(s)}</p></div>', unsafe_allow_html=True)
def head(i,h): st.markdown(f'<div class="head"><div class="ic">{i}</div><h3>{escape(h)}</h3></div>', unsafe_allow_html=True)
def fmt(x,d=1):
    try: n=float(x)
    except Exception: return "--"
    return f"{n:.0f}" if abs(n-round(n))<1e-8 else f"{n:.{d}f}"
def pct(x):
    try: return f"{float(x):.0%}"
    except Exception: return "--"
def fsh_cat(d):
    if d<80:return("0-80","预测")
    if d<=160:return("80-160","预测")
    return(">160","预测")
def lh_cat(d): return ("0","预测") if d<=0 else ("75","预测") if d<=75 else (">75","预测")
def hmg_cat(d): return ("0","预测") if d<=0 else ("75","预测") if d<=75 else ("150","预测") if d<=150 else (">150","预测")
def dose_class(drug, dose):
    return {"fsh":fsh_cat, "lh":lh_cat, "hmg":hmg_cat}[drug](float(dose))[0]
def display_dose_category(drug, value):
    text=str(value).strip()
    key=text.lower().replace("_","-").replace(" ","")
    if key.endswith("iu"):
        key=key[:-2]
    aliases={
        "fsh":{"low/none":"0-80","low-none":"0-80","lownone":"0-80","low":"0-80","0-80":"0-80","moderate":"80-160","80-160":"80-160","high":">160",">160":">160"},
        "lh":{"dose-0":"0","dose0":"0","0":"0","dose-75":"75","dose75":"75","75":"75","dose-150":"150","dose150":"150","dose-150-plus":"150","dose150plus":"150","150":"150",">75":"150"},
        "hmg":{"dose-0":"0","dose0":"0","0":"0","dose-75":"75","dose75":"75","75":"75","dose-150":"150","dose150":"150","150":"150","dose-225":"225","dose225":"225","dose-225-plus":"225","dose225plus":"225","225":"225",">150":"225"},
    }
    if key in aliases[drug]:
        return aliases[drug][key]
    try:
        return dose_class(drug,float(text))
    except Exception:
        return text
def pred_cell(label): return f'{escape(str(label))}<span class="pred-note">预测</span>'
def field_label(container, text, required=False):
    star='<span class="req">*</span>' if required else ''
    container.markdown(f'<div class="field-label">{escape(text)} {star}</div>', unsafe_allow_html=True)
def dash_dose(container, text):
    field_label(container, text, False)
    container.markdown('<div class="dash-dose"><b>-</b><span>由模型预测</span></div>', unsafe_allow_html=True)
def parse_date(value):
    try: return datetime.strptime(str(value), "%Y-%m-%d").date()
    except Exception:
        try: return datetime.strptime(str(value), "%Y/%m/%d").date()
        except Exception: return datetime.now().date()
DOSE_KEYS=("current_fsh","current_lh","current_hmg")
FOLLICLE_BIN_KEYS=("f_lt10","f_10_12","f_13_15","f_16_18","f_gt18")
def as_float(value, default=0.0):
    try: return float(value)
    except Exception: return float(default)
def bound_predicted_counts(oocytes, mii, total_follicles):
    o=clamp(as_float(oocytes),0,60)
    m=clamp(as_float(mii),0,60)
    cap=as_float(total_follicles,-1)
    if cap>=0:
        o=min(o,cap)
    m=min(m,o)
    return o,m
def has_follicle_bins(record):
    return any(key in record for key in FOLLICLE_BIN_KEYS) if isinstance(record, Mapping) else False
def derived_total_follicles(record):
    if not isinstance(record, Mapping): return 0.0
    binned=sum(as_float(record.get(key),0.0) for key in FOLLICLE_BIN_KEYS)
    explicit=record.get("total_follicles", record.get("total_follicle_count"))
    if explicit not in (None, ""):
        return as_float(explicit, binned)
    return binned
def normalize_monitoring_record(record):
    item=dict(record) if isinstance(record, Mapping) else {}
    item["total_follicles"]=int(round(derived_total_follicles(item)))
    return item
MONITORING_RECORD_KEYS=("visit","stim_day","days_since_previous_visit","monitoring_date","e2","lh_value","p","serum_fsh","current_endometrium","current_fsh","current_lh","current_hmg","total_follicles","left_follicles","right_follicles","max_f","mean_f","f_lt10","f_10_12","f_13_15","f_16_18","f_gt18")
REALISTIC_MONITORING_TEMPLATES={
    1:dict(stim_day=1,days_since_previous_visit=0,e2=42.0,lh_value=4.8,p=0.50,serum_fsh=7.2,current_endometrium=5.6,total_follicles=12,left_follicles=6,right_follicles=6,f_lt10=12,f_10_12=0,f_13_15=0,f_16_18=0,f_gt18=0,max_f=8.5,mean_f=5.8,_demo_template="realistic_default_case",_visit_note="start_stimulation"),
    2:dict(stim_day=5,days_since_previous_visit=4,e2=486.0,lh_value=2.6,p=0.42,serum_fsh=8.9,current_endometrium=7.4,total_follicles=14,left_follicles=7,right_follicles=7,f_lt10=6,f_10_12=5,f_13_15=3,f_16_18=0,f_gt18=0,max_f=14.0,mean_f=10.4,_demo_template="realistic_default_case",_visit_note="early_response"),
    3:dict(stim_day=8,days_since_previous_visit=3,e2=1580.0,lh_value=2.8,p=0.62,serum_fsh=7.6,current_endometrium=10.0,total_follicles=16,left_follicles=8,right_follicles=8,f_lt10=5,f_10_12=5,f_13_15=3,f_16_18=2,f_gt18=1,max_f=18.0,mean_f=12.6,_demo_template="realistic_default_case",_visit_note="pre_trigger_assessment"),
    4:dict(stim_day=10,days_since_previous_visit=2,e2=2860.0,lh_value=2.4,p=0.86,serum_fsh=6.9,current_endometrium=11.2,total_follicles=17,left_follicles=9,right_follicles=8,f_lt10=2,f_10_12=3,f_13_15=4,f_16_18=5,f_gt18=3,max_f=21.0,mean_f=15.4,_demo_template="realistic_default_case",_visit_note="trigger_day"),
}
if REPRESENTATIVE_CASE.get("monitoring_records"):
    REALISTIC_MONITORING_TEMPLATES={
        index + 1: dict(record)
        for index, record in enumerate(REPRESENTATIVE_CASE["monitoring_records"])
        if isinstance(record, Mapping)
    }
def _monitoring_date_for_stim_day(stim_day):
    try:
        offset=max(0, int(stim_day)-1)
    except Exception:
        offset=0
    return str(parse_date(DEFAULT.get("monitoring_date")) + timedelta(days=offset))
def _generated_monitoring_template(visit, stim_day=None):
    visit_int=max(1, int(visit))
    if visit_int in REALISTIC_MONITORING_TEMPLATES:
        template=dict(REALISTIC_MONITORING_TEMPLATES[visit_int])
    else:
        template=dict(REALISTIC_MONITORING_TEMPLATES[3])
        extra=max(0, visit_int-3)
        base_e2=as_float(template.get("e2"),0.0)
        base_endometrium=as_float(template.get("current_endometrium"),0.0)
        base_total=int(round(as_float(template.get("total_follicles"),0.0)))
        base_left=int(round(as_float(template.get("left_follicles"),0.0)))
        base_right=int(round(as_float(template.get("right_follicles"),0.0)))
        base_max=as_float(template.get("max_f"),0.0)
        base_mean=as_float(template.get("mean_f"),0.0)
        template.update({
            "stim_day":int(template.get("stim_day",8)) + extra * 2,
            "days_since_previous_visit":2,
            "e2":min(4200.0, base_e2 + extra * max(300.0, base_e2 * .30)),
            "p":min(2.5, as_float(template.get("p"),0.0) + extra * 0.18),
            "current_endometrium":min(14.0, base_endometrium + extra * 0.8),
            "total_follicles":min(24, base_total + extra),
            "left_follicles":min(12, base_left + (extra + 1)//2),
            "right_follicles":min(12, base_right + extra//2),
            "f_lt10":max(1, int(template.get("f_lt10",0)) - extra),
            "f_10_12":max(1, int(template.get("f_10_12",0)) - extra),
            "f_13_15":min(6, int(template.get("f_13_15",0)) + extra),
            "f_16_18":min(7, int(template.get("f_16_18",0)) + extra),
            "f_gt18":min(6, int(template.get("f_gt18",0)) + extra),
            "max_f":min(24.0, base_max + extra * 1.8),
            "mean_f":min(17.5, base_mean + extra * 1.0),
        })
    if stim_day is not None:
        template["stim_day"]=int(stim_day)
    template["monitoring_date"]=_monitoring_date_for_stim_day(template.get("stim_day", visit_int))
    return template
def default_monitoring_record(visit=1, stim_day=None, days_since_previous_visit=None):
    template=_generated_monitoring_template(visit, stim_day)
    record={key: DEFAULT.get(key) for key in MONITORING_RECORD_KEYS}
    record.update(template)
    record["visit"]=int(visit)
    record["stim_day"]=int(template.get("stim_day", visit))
    if days_since_previous_visit is not None:
        record["days_since_previous_visit"]=int(days_since_previous_visit)
    elif int(record["visit"]) <= 1:
        record["days_since_previous_visit"]=0
    else:
        record["days_since_previous_visit"]=int(template.get("days_since_previous_visit", 3))
    record["monitoring_date"]=_monitoring_date_for_stim_day(record["stim_day"])
    return normalize_monitoring_record(record)
def default_monitoring_records():
    source=REPRESENTATIVE_CASE.get("monitoring_records")
    if not isinstance(source, list) or not source:
        return [default_monitoring_record()]
    records=[]
    for index, raw in enumerate(source):
        if not isinstance(raw, Mapping):
            continue
        record=default_monitoring_record(index + 1)
        record.update(dict(raw))
        record["visit"]=index + 1
        records.append(normalize_monitoring_record(record))
    if not records:
        return [default_monitoring_record()]
    for key in DOSE_KEYS:
        records[-1].pop(key, None)
    return records
def next_monitoring_timing(records):
    clean=[r for r in records if isinstance(r, Mapping)] if isinstance(records, list) else []
    next_visit=max([int(r.get("visit", i+1)) for i, r in enumerate(clean)] or [0]) + 1
    prev_stim=max([int(r.get("stim_day", i+1)) for i, r in enumerate(clean)] or [0])
    template=_generated_monitoring_template(next_visit)
    if next_visit in REALISTIC_MONITORING_TEMPLATES:
        target_stim=int(template.get("stim_day", prev_stim + 3))
    else:
        target_stim=max(prev_stim + 2, int(template.get("stim_day", prev_stim + 2)))
    target_stim=min(30, target_stim)
    gap=max(1, target_stim - prev_stim) if clean else 0
    return next_visit, target_stim, gap
def scrub_latest_dose_targets(records):
    clean=[normalize_monitoring_record(r) for r in records if isinstance(r, Mapping)] if isinstance(records, list) else []
    if not clean:
        clean=default_monitoring_records()
    for key in DOSE_KEYS:
        clean[-1].pop(key, None)
    return clean
def monitoring_records():
    records=st.session_state.setdefault("monitoring_records", default_monitoring_records())
    if not isinstance(records, list) or not records:
        records=default_monitoring_records()
    records=scrub_latest_dose_targets(records)
    st.session_state.monitoring_records=records
    return records
def reference_dose_context(patient, records):
    history=[r for r in (records[:-1] if records else []) if isinstance(r, dict) and any(k in r for k in DOSE_KEYS)]
    if history:
        src=history[-1]
        fsh=as_float(src.get("current_fsh"), 0.0)
        lh=as_float(src.get("current_lh"), 0.0)
        hmg=as_float(src.get("current_hmg"), 0.0)
        source="previous monitoring executed dose"
    else:
        fsh=0.0
        lh=0.0
        hmg=0.0
        source="no historical executed dose"
    return {"reference_fsh":fsh,"reference_lh":lh,"reference_hmg":hmg,"current_fsh":fsh,"current_lh":lh,"current_hmg":hmg,"previous_gn_dose":fsh+lh+hmg,"reference_dose_source":source}
def patient_with_latest_monitoring(patient, records):
    records=[normalize_monitoring_record(r) for r in records] if records else []
    merged=dict(patient)
    if records:
        merged.update(records[-1])
    merged.update(reference_dose_context(merged, records))
    return merged

def temporal_history_snapshots(patient, records):
    if patient_form_to_snapshot is None:
        return []
    clean=[normalize_monitoring_record(r) for r in (records or []) if isinstance(r, Mapping)]
    snapshots=[]
    for index, record in enumerate(clean):
        form=dict(patient or {})
        form.update(record)
        previous=clean[index-1] if index else {}
        form["reference_fsh"]=as_float(previous.get("current_fsh"), as_float(form.get("initial_fsh"), 0.0))
        form["reference_lh"]=as_float(previous.get("current_lh"), as_float(form.get("initial_lh"), 0.0))
        form["reference_hmg"]=as_float(previous.get("current_hmg"), as_float(form.get("initial_hmg"), 0.0))
        try:
            snapshots.append(patient_form_to_snapshot(form))
        except Exception:
            continue
    return snapshots

def _as_prob(value):
    try:
        value=float(value)
    except Exception:
        return None
    if value != value:
        return None
    return clamp(value,0.0,1.0)

def risk_label(prob, low=None, high=None, category=None):
    prob=_as_prob(prob)
    if prob is None:
        return ("\u672a\u8fd4\u56de","cm","Unknown")
    low=float(OHSS_RISK_DEFAULTS["threshold_low"] if low is None else low)
    high=float(OHSS_RISK_DEFAULTS["threshold_high"] if high is None else high)
    if category:
        key=str(category).strip().lower()
        if key in ("high","\u9ad8\u98ce\u9669"):
            return ("\u9ad8\u98ce\u9669","cd","High")
        if key in ("moderate","medium","\u4e2d\u98ce\u9669"):
            return ("\u4e2d\u98ce\u9669","cw","Moderate")
        if key in ("low","\u4f4e\u98ce\u9669"):
            return ("\u4f4e\u98ce\u9669","ct","Low")
    return ("\u9ad8\u98ce\u9669","cd","High") if prob>=high else ("\u4e2d\u98ce\u9669","cw","Moderate") if prob>=low else ("\u4f4e\u98ce\u9669","ct","Low")

def _ohss_reference_probabilities():
    path=REPO_ROOT / str(OHSS_RISK_DEFAULTS["reference_predictions_path"])
    if not path.exists(): return []
    values=[]
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                value=_as_prob(row.get("calibrated_probability"))
                if value is not None: values.append(value)
    except Exception:
        return []
    return values

def ohss_percentile(prob, explicit=None):
    if explicit is not None:
        try: return int(round(clamp(float(explicit),0,100)))
        except Exception: pass
    prob=_as_prob(prob)
    if prob is None: return None
    refs=_ohss_reference_probabilities()
    if not refs: return None
    return int(round(sum(v <= prob for v in refs) * 100 / len(refs)))

def ohss_profile(row):
    prob=_as_prob(row.get("ohss_risk_probability", row.get("safety_probability"))) if isinstance(row,Mapping) else None
    low=row.get("ohss_risk_threshold_low") if isinstance(row,Mapping) else None
    high=row.get("ohss_risk_threshold_high") if isinstance(row,Mapping) else None
    category=row.get("ohss_risk_category") if isinstance(row,Mapping) else None
    zh,cls,en=risk_label(prob,low,high,category)
    percentile=ohss_percentile(prob, row.get("ohss_risk_percentile") if isinstance(row,Mapping) else None)
    return dict(prob=prob,category_zh=zh,category_en=en,cls=cls,percentile=percentile,threshold_low=low or OHSS_RISK_DEFAULTS["threshold_low"],threshold_high=high or OHSS_RISK_DEFAULTS["threshold_high"])

def ohss_display_profile(row):
    enriched=with_ohss_display_fields(row if isinstance(row,Mapping) else {})
    raw=_as_prob(enriched.get("ohss_raw_probability", enriched.get("ohss_risk_probability", enriched.get("safety_probability"))))
    display=_as_prob(enriched.get("ohss_display_probability"))
    if display is None:
        display=raw
    return dict(raw=raw, display=display, label=enriched.get("ohss_display_label", "模型原始概率"), note=enriched.get("ohss_display_note", ""))

def ohss_display_pct(row):
    display=ohss_display_profile(row).get("display")
    return f"{display * 100:.2f}%" if display is not None else "--"

def ohss_raw_probability_note(row):
    raw=ohss_display_profile(row).get("raw")
    return "\u4e25\u683c\u4e2d\u91cd\u5ea6 OHSS \u6a21\u578b\u539f\u59cb\u6821\u51c6\u6982\u7387" if raw is not None else "\u4e2d\u91cd\u5ea6 OHSS \u98ce\u9669 --"

def risk(prob):
    prof=ohss_profile({"ohss_risk_probability":prob})
    return prof["category_zh"], prof["cls"]

def round_step(x,step,mx): return round(clamp(x,0,mx)/step)*step
def evidence(action): return {"increase":dict(selection=.28, ovarian=.72), "decrease":dict(selection=.18, ovarian=.68)}.get(action, dict(selection=.54, ovarian=.76))

def _dose_model_plan_candidates(ctx, cf, cl, ch):
    plans=[("\u5f53\u524d\u8bb0\u5f55\u65b9\u6848","\u6bd4\u8f83\u57fa\u51c6",cf,cl,ch,"current")]
    preds=(ctx or {}).get("predictions") if isinstance(ctx,Mapping) else None
    if not isinstance(preds,Mapping):
        plans.append(("\u6a21\u578b\u6682\u4e0d\u53ef\u7528","UI-reduced model unavailable",cf,cl,ch,"recommended"))
        return plans
    rec={drug:as_float(preds[drug].get("dose"),0.0) for drug in ("fsh","lh","hmg") if isinstance(preds.get(drug),Mapping)}
    for drug, fallback in (("fsh",cf),("lh",cl),("hmg",ch)):
        rec.setdefault(drug, fallback)
    plans.append(("\u5019\u9009 1","UI-reduced GRU(AddGate) \u9996\u9009",rec["fsh"],rec["lh"],rec["hmg"],"recommended"))
    drug_names={"fsh":"FSH","lh":"LH","hmg":"HMG"}
    rank=2
    for drug in ("fsh","lh","hmg"):
        top=list((preds.get(drug) or {}).get("top_labels") or [])
        for alt in top[1:3]:
            combo=dict(rec); combo[drug]=as_float(alt.get("dose"),combo[drug])
            plans.append((f"\u5019\u9009 {rank}",f"{drug_names[drug]} \u6b21\u9ad8\u6982\u7387\u7c7b\u522b",combo["fsh"],combo["lh"],combo["hmg"],"candidate")); rank+=1
    fsh_levels=[40.0,120.0,200.0]
    hmg_levels=[0.0,75.0,150.0,225.0]
    lh_levels=[0.0,75.0,150.0]
    def lower(value, levels):
        lower_values=[x for x in levels if x < value]
        return lower_values[-1] if lower_values else value
    def higher(value, levels):
        higher_values=[x for x in levels if x > value]
        return higher_values[0] if higher_values else value
    plans.append((f"\u5019\u9009 {rank}","\u5b89\u5168\u4fdd\u5b88\u7ec4\u5408",lower(rec["fsh"],fsh_levels),lower(rec["lh"],lh_levels),lower(rec["hmg"],hmg_levels),"candidate")); rank+=1
    plans.append((f"\u5019\u9009 {rank}","\u53cd\u5e94\u589e\u5f3a\u7ec4\u5408",higher(rec["fsh"],fsh_levels),higher(rec["lh"],lh_levels),higher(rec["hmg"],hmg_levels),"candidate"))
    unique=[]; seen=set()
    for name,tag,f,l,h,role in plans:
        key=(round(f,3),round(l,3),round(h,3),role)
        if key in seen and role!="recommended":
            continue
        seen.add(key); unique.append((name,tag,f,l,h,role))
    return unique

def _dose_response_curve_plan_candidates(base_fsh, base_lh, base_hmg):
    """Return formal recommendation candidates sampled from dose-response curves.

    These are the same kind of single-axis dose perturbations shown in the KNN
    dose-response cards. The final UI recommendation is selected from these
    points by a gate-free predicted-oocyte/OHSS Pareto balance, while every
    candidate remains eligible for clinician review.
    """
    levels = {
        "fsh": [40.0, 120.0, 200.0],
        "lh": [0.0, 75.0, 150.0],
        "hmg": [0.0, 75.0, 150.0, 225.0],
    }
    base = {"fsh": as_float(base_fsh, 0.0), "lh": as_float(base_lh, 0.0), "hmg": as_float(base_hmg, 0.0)}
    labels = {"fsh": "FSH", "lh": "LH", "hmg": "HMG"}
    out = []
    seen = set()
    rank = 1
    for axis in ("fsh", "lh", "hmg"):
        for dose in levels[axis]:
            combo = dict(base)
            combo[axis] = dose
            key = (round(combo["fsh"], 3), round(combo["lh"], 3), round(combo["hmg"], 3))
            if key in seen:
                continue
            seen.add(key)
            out.append(
                dict(
                    name=f"\u66f2\u7ebf\u5019\u9009 {rank}",
                    tag=f"{labels[axis]} \u5242\u91cf-\u53cd\u5e94\u66f2\u7ebf情景点",
                    f=combo["fsh"],
                    l=combo["lh"],
                    h=combo["hmg"],
                    role="candidate",
                    candidate_family="dose_response_curve",
                    curve_axis=axis,
                    recommendation_basis="response-stratified dose-response scenario",
                )
            )
            rank += 1
    return out

def recommend(v:Mapping[str,Any])->list[dict[str,Any]]:
    age=float(v["age"]); amh=float(v["amh"]); afc=float(v["afc"]); e2=float(v["e2"]); cf=as_float(v.get("reference_fsh",v.get("current_fsh")),0.0); cl=as_float(v.get("reference_lh",v.get("current_lh")),0.0); ch=as_float(v.get("reference_hmg",v.get("current_hmg")),0.0); mat=float(v.get("f_16_18",0))+float(v.get("f_gt18",0)); grow=float(v.get("f_10_12",0))+float(v.get("f_13_15",0)); mf=float(v["max_f"])
    snap=None; layer_ctx=None; dose_ctx=None
    records=[normalize_monitoring_record(r) for r in st.session_state.get("monitoring_records", []) if isinstance(r, Mapping)] or default_monitoring_records()
    history_snapshots=temporal_history_snapshots(v, records)
    if LAYER1_AVAILABLE:
        try:
            snap=patient_form_to_snapshot(v); layer_ctx=predict_layer1_action_context(snap,k=50)
            st.session_state["layer1_context"]=layer_ctx
            st.session_state["layer1_context_at"]=datetime.now().isoformat(timespec="seconds")
        except Exception as exc: st.session_state["layer1_error"]=str(exc)
    if snap is None and patient_form_to_snapshot is not None:
        try: snap=patient_form_to_snapshot(v)
        except Exception as exc: st.session_state["snapshot_error"]=str(exc)
    if DOSE_RECOMMENDATION_AVAILABLE and predict_ui_reduced_dose_context is not None:
        try:
            dose_ctx=predict_ui_reduced_dose_context(v, records)
            st.session_state["dose_recommendation_context"]=dose_ctx
            st.session_state["dose_recommendation_warnings"]=list(dose_ctx.get("warnings") or [])
            st.session_state.pop("dose_recommendation_error", None)
        except Exception as exc:
            st.session_state["dose_recommendation_error"]=str(exc)
            st.session_state["dose_recommendation_warnings"]=[]
    else:
        st.session_state["dose_recommendation_error"]=globals().get("DOSE_RECOMMENDATION_ERROR","UI-reduced dose model import failed")
    dose_model_plans=_dose_model_plan_candidates(dose_ctx, cf, cl, ch)
    current_plan=next((plan for plan in dose_model_plans if plan[5] == "current"), ("当前记录方案","比较基准",cf,cl,ch,"current"))
    classifier_anchor=next((plan for plan in dose_model_plans if plan[5] == "recommended"), current_plan)
    _,_,anchor_fsh,anchor_lh,anchor_hmg,_=classifier_anchor
    plan_specs=[dict(name=current_plan[0],tag=current_plan[1],f=current_plan[2],l=current_plan[3],h=current_plan[4],role="current",candidate_family="current",curve_axis="",recommendation_basis="current executed-dose reference")]
    plan_specs.extend(_dose_response_curve_plan_candidates(anchor_fsh, anchor_lh, anchor_hmg))
    out=[]; cur=cf+cl+ch
    for plan in plan_specs:
        name=plan["name"]; tag=plan["tag"]; f=plan["f"]; l=plan["l"]; h=plan["h"]; role=plan["role"]
        candidate_family=plan.get("candidate_family","dose_model"); curve_axis=plan.get("curve_axis","")
        f=clamp(as_float(f),0,600); l=round_step(l,37.5,300); h=round_step(h,37.5,450); total=f+l+h
        action="increase" if total-cur>=37.5 else "decrease" if total-cur<=-37.5 else "maintain"; ev=evidence(action)
        if layer_ctx is not None and evidence_for_action is not None:
            try:
                real_ev=evidence_for_action(layer_ctx,action)
                sel=float(real_ev.get("selection",ev["selection"]))
                ovar=float(real_ev.get("ovarian",ev["ovarian"]))
                if sel==sel: ev["selection"]=sel
                if ovar==ovar: ev["ovarian"]=ovar
            except Exception as exc:
                st.session_state["layer1_evidence_error"]=str(exc)
        reserve=.55*clamp(amh/3,0,2)+.45*clamp(afc/12,0,2); follicle=mat*.78+grow*.35+mf*.06; o=clamp(2.2+reserve*2.7+follicle+(total-cur)/150*1.4-max(age-34,0)*.13,1,32); m=clamp(o*clamp(.66+.012*mf-.008*max(age-35,0)+.025*(l>0 or h>0),.5,.88),0,o)
        prob=None; score=None; scoring_source="formula_fallback"
        ohss_contributors=[]; ohss_source="formula_fallback"; ohss_disclaimer=OHSS_UI_DISCLAIMER
        if CANDIDATE_RESPONSE_AVAILABLE and snap is not None and score_candidate_response is not None:
            try:
                real=score_candidate_response(
                    snap,
                    fsh_dose=f,
                    lh_dose=l,
                    hmg_dose=h,
                    history_snapshots=history_snapshots,
                )
                o=clamp(float(real["oocytes"]),0,60)
                m=clamp(float(real["mii"]),0,60)
                scoring_source=str(real.get("model_name", real.get("source","candidate_response_ui_best_v2")))
                prob=clamp(float(real.get("ohss_risk", real.get("ohss_risk_raw", 0.0))),0,1)
                score=round(prob*100)
                ohss_source=str(real.get("ohss_run_id") or real.get("ohss_model_name") or real.get("run_id") or real.get("source") or "candidate_response_layer2_ohss")
            except Exception as exc:
                st.session_state["candidate_response_error"]=str(exc)
        if OUTCOME_AVAILABLE and snap is not None and not CANDIDATE_RESPONSE_AVAILABLE:
            try:
                real=score_effectiveness_safety_outcomes(snap,fsh_dose=f,lh_dose=l,hmg_dose=h)
                o=clamp(float(real["oocytes"]),0,60)
                m=clamp(float(real["mii"]),0,60)
                scoring_source=str(real.get("source","real_effectiveness_safety_bundles"))
            except Exception as exc:
                st.session_state["outcome_error"]=str(exc)
        if prob is None:
            prob=clamp(.08+max(e2-1200,0)/6500+max(mat-5,0)*.018+max(total-cur,0)/3500,.04,.72)
            score=round(prob*100)
            ohss_source="formula_fallback_strict_endpoint_unavailable"
        o,m=bound_predicted_counts(o,m,v.get("total_follicles", v.get("total_follicle_count")))
        tmp=dict(ohss_risk_probability=prob,ohss_risk_threshold_low=OHSS_RISK_DEFAULTS["threshold_low"],ohss_risk_threshold_high=OHSS_RISK_DEFAULTS["threshold_high"])
        prof=ohss_profile(tmp)
        row=dict(
            name=name,tag=tag,candidate_role=role,action=action,current_total=cur,candidate_total=total,
            fsh=f,lh=l,hmg=h,o=round(o,1),o_selection_value=float(o),mii=round(m,1),mii_selection_value=float(m),
            safety_probability=prof["prob"],strict_ohss_probability=prof["prob"],safety_score=score,
            ohss_risk_probability=prof["prob"],ohss_risk_category=prof["category_en"],
            ohss_risk_percentile=prof["percentile"],ohss_risk_threshold_low=prof["threshold_low"],
            ohss_risk_threshold_high=prof["threshold_high"],ohss_contributors=ohss_contributors,
            ohss_source=ohss_source,ohss_disclaimer=ohss_disclaimer,selection_rate=ev["selection"],
            success_rate=ev["ovarian"],score=round(o,3),fsh_category=fsh_cat(f),lh_category=lh_cat(l),
            hmg_category=hmg_cat(h),dose_model_source=(dose_ctx or {}).get("source","ui_reduced_model_unavailable"),
            dose_model_context=dose_ctx,scoring_source=scoring_source,candidate_family=candidate_family,
            curve_axis=curve_axis,recommendation_basis=plan.get("recommendation_basis","dose-response curve scenario"),
        )
        row.update(with_ohss_display_fields(row))
        out.append(row)
    if CANDIDATE_BALANCE_AVAILABLE and apply_oocyte_ohss_balance_recommendation is not None:
        try:
            out=apply_oocyte_ohss_balance_recommendation(out)
            selected=next((row for row in out if row.get("candidate_role")=="recommended"),None)
            st.session_state.pop("ovarian_response_context",None)
            st.session_state["candidate_balance_context"]={
                "source":"candidate_oocyte_modsev_ohss_balance_no_gate_v1",
                "basis":selected.get("recommendation_basis") if selected else "oocyte/OHSS Pareto balance",
            }
            st.session_state.pop("candidate_balance_error",None)
            return out
        except Exception as exc:
            st.session_state["candidate_balance_error"]=str(exc)
    else:
        st.session_state["candidate_balance_error"]=globals().get(
            "CANDIDATE_BALANCE_ERROR",
            "Oocyte/OHSS balance selector import failed",
        )
    if not any(x.get("candidate_role")=="recommended" for x in out):
        candidates=[x for x in out if x.get("candidate_role")!="current"]
        if candidates: candidates[0]["candidate_role"]="recommended"
    return out

def recommendation_signature(patient, records):
    def clean(value):
        if isinstance(value, Mapping):
            return {str(k): clean(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value
    return json.dumps({"patient": clean(patient), "records": clean(records)}, ensure_ascii=False, sort_keys=True, default=str)

def _selected_recommendation_row(rows):
    items=[r for r in (rows or []) if isinstance(r, Mapping)]
    for row in items:
        if row.get("candidate_role") == "recommended":
            return row
    candidates=[r for r in items if r.get("candidate_role") != "current"]
    return max(candidates or items, key=lambda r: as_float(r.get("score"), -1.0), default=None)

def recommendation_input_patient(patient, records):
    model_patient=dict(patient)
    clean_records=[r for r in (records or []) if isinstance(r, Mapping)]
    if len(clean_records) == 1:
        model_patient.update({
            "initial_fsh":0.0,
            "initial_lh":0.0,
            "initial_hmg":0.0,
            "initial_gn":0.0,
            "initial_gn_dose":0.0,
            "initial_gn_source":"pending_first_monitoring_prediction",
        })
    return model_patient

def apply_first_prediction_as_initial_gn(patient, records, rows):
    clean_records=[r for r in (records or []) if isinstance(r, Mapping)]
    if len(clean_records) != 1:
        return patient
    row=_selected_recommendation_row(rows)
    if row is None:
        return patient
    fsh=as_float(row.get("fsh"),0.0)
    lh=as_float(row.get("lh"),0.0)
    hmg=as_float(row.get("hmg"),0.0)
    updated=dict(patient)
    updated.update({
        "initial_fsh":fsh,
        "initial_lh":lh,
        "initial_hmg":hmg,
        "initial_gn":fsh+lh+hmg,
        "initial_gn_dose":fsh+lh+hmg,
        "initial_gn_source":"first_monitoring_prediction",
    })
    return updated

def executed_dose_defaults_from_prior_decision(base_patient, records, allow_recompute=False):
    clean_records=[normalize_monitoring_record(r) for r in records if isinstance(r, Mapping)] or default_monitoring_records()
    patient=patient_with_latest_monitoring(base_patient, clean_records)
    sig=recommendation_signature(patient, clean_records)
    rows=st.session_state.get("recs") if st.session_state.get("recs_signature") == sig else None
    if rows is None and allow_recompute:
        saved_records=st.session_state.get("monitoring_records")
        saved_patient=st.session_state.get("patient")
        try:
            st.session_state.monitoring_records=clean_records
            st.session_state.patient=patient
            rows=recommend(patient)
            st.session_state.recs=rows
            st.session_state.recs_signature=sig
            st.session_state.recs_patient=patient
        except Exception as exc:
            st.session_state["executed_dose_prefill_error"]=str(exc)
            rows=None
        finally:
            if saved_records is not None:
                st.session_state.monitoring_records=saved_records
            if saved_patient is not None:
                st.session_state.patient=saved_patient
    row=_selected_recommendation_row(rows)
    if row is not None:
        return {"current_fsh":as_float(row.get("fsh"),0.0),"current_lh":as_float(row.get("lh"),0.0),"current_hmg":as_float(row.get("hmg"),0.0),"_executed_dose_default_source":"previous_ai_recommendation"}
    ctx=reference_dose_context(patient, clean_records)
    return {"current_fsh":as_float(ctx.get("reference_fsh"),0.0),"current_lh":as_float(ctx.get("reference_lh"),0.0),"current_hmg":as_float(ctx.get("reference_hmg"),0.0),"_executed_dose_default_source":"reference_dose_fallback"}

def mark_latest_record_as_executed(base_patient, records, allow_recompute=False):
    clean_records=[normalize_monitoring_record(r) for r in records if isinstance(r, Mapping)] or default_monitoring_records()
    defaults=executed_dose_defaults_from_prior_decision(base_patient, clean_records, allow_recompute=allow_recompute)
    for key in DOSE_KEYS:
        clean_records[-1][key]=defaults[key]
    clean_records[-1]["_executed_dose_default_source"]=defaults.get("_executed_dose_default_source", "reference_dose_fallback")
    return clean_records

def persist_patient_records(base_patient, records, recompute=True):
    clean_records=scrub_latest_dose_targets(records)
    patient=patient_with_latest_monitoring(base_patient, clean_records)
    st.session_state.monitoring_records=clean_records
    st.session_state.patient=patient
    sig=recommendation_signature(patient, clean_records)
    if recompute:
        model_patient=recommendation_input_patient(patient, clean_records)
        st.session_state.patient=model_patient
        rows=recommend(model_patient)
        patient=apply_first_prediction_as_initial_gn(patient, clean_records, rows)
        st.session_state.patient=patient
        sig=recommendation_signature(patient, clean_records)
        st.session_state.recs=rows
        st.session_state.recs_signature=sig
        st.session_state.recs_patient=patient
        st.session_state.recs_stale=False
        st.session_state.pop("pending_recs_signature", None)
    else:
        st.session_state.recs_stale=bool(st.session_state.get("recs"))
        st.session_state.pending_recs_signature=sig
    return patient

def sync_recommendations(force=False, show_status=False, page_label=None, auto_recompute=True):
    records=[normalize_monitoring_record(r) for r in monitoring_records()]
    patient=patient_with_latest_monitoring(st.session_state.get("patient", DEFAULT.copy()), records)
    sig=recommendation_signature(patient, records)
    has_recs=bool(st.session_state.get("recs"))
    needs_refresh=force or not has_recs or st.session_state.get("recs_signature") != sig
    if needs_refresh:
        st.session_state.monitoring_records=records
        st.session_state.patient=patient
        if not force and not auto_recompute:
            st.session_state.recs_stale=has_recs
            st.session_state.pending_recs_signature=sig
            return st.session_state.get("recs_patient", patient) if has_recs else patient
        label=escape(str(page_label or "当前页面"))
        model_patient=recommendation_input_patient(patient, records)
        st.session_state.patient=model_patient
        if show_status:
            st.markdown(f'<div class="model-running"><span class="model-dot running"></span><span>{label} 正在生成最新推荐结果。</span></div>', unsafe_allow_html=True)
            with st.spinner("模型计算中"):
                rows=recommend(model_patient)
        else:
            rows=recommend(model_patient)
        patient=apply_first_prediction_as_initial_gn(patient, records, rows)
        sig=recommendation_signature(patient, records)
        st.session_state.patient=patient
        st.session_state.recs=rows
        st.session_state.recs_signature=sig
        st.session_state.recs_patient=patient
        st.session_state.recs_stale=False
        st.session_state.pop("pending_recs_signature", None)
    return st.session_state.patient

def stale_recommendation_notice():
    if st.session_state.get("recs_stale"):
        st.markdown('<div class="mini-hint">当前页面使用上一次已生成的推荐结果；患者录入已有新修改，请在录入页点击“生成监测结果”后再刷新此页。</div>', unsafe_allow_html=True)

def refresh_page_recommendations(page_label):
    stale=bool(st.session_state.get("recs_stale"))
    return sync_recommendations(
        show_status=stale,
        page_label=page_label,
        auto_recompute=stale,
    )

def recommendation_required_notice(page_label):
    if st.session_state.get("recs"):
        return True
    label=escape(str(page_label or "当前页面"))
    st.markdown(f'<div class="warning">{label} 尚未生成监测结果。请先在患者录入页点击“生成监测结果”，或在此处点击下方按钮生成最新监测结果。</div>', unsafe_allow_html=True)
    key="generate_missing_recs_" + str(page_label or "page")
    if st.button("生成最新监测结果", type="primary", use_container_width=True, key=key):
        sync_recommendations(force=True, show_status=True, page_label=page_label)
        rerun()
    return False

def best(auto_recompute=True):
    sync_recommendations(auto_recompute=auto_recompute)
    return next((x for x in st.session_state.recs if x.get("candidate_role")=="recommended"), max(st.session_state.recs,key=lambda x:x["score"]))

def current_layer1_context():
    records=[normalize_monitoring_record(r) for r in monitoring_records()]
    patient=patient_with_latest_monitoring(st.session_state.get("patient", DEFAULT.copy()), records)
    sig=recommendation_signature(patient, records)
    ctx=st.session_state.get("layer1_context")
    if ctx and st.session_state.get("layer1_context_signature") == sig:
        return ctx
    if not LAYER1_AVAILABLE:
        return None
    try:
        snap=patient_form_to_snapshot(patient)
        ctx=predict_layer1_action_context(snap,k=50)
        st.session_state["layer1_context"]=ctx
        st.session_state["layer1_context_signature"]=sig
        st.session_state["layer1_context_at"]=datetime.now().isoformat(timespec="seconds")
        st.session_state.pop("layer1_error", None)
        return ctx
    except Exception as exc:
        st.session_state["layer1_error"]=str(exc)
        return None

def home():
    title("医生端临床辅助工作台","围绕 IVF/ICSI 促排监测、下一次记录性 Gn 剂量预测、相似决策点与可解释 AI 的低噪声工作界面。")
    st.markdown('<div class="grid4"><div class="metric"><div class="ml">今日待评估患者</div><div class="mv">12</div><div class="note">较昨日 +3</div></div><div class="metric"><div class="ml">已生成推荐</div><div class="mv teal">8</div><div class="note">均待医生最终确认</div></div><div class="metric"><div class="ml">中重度预警待复核</div><div class="mv warn">2</div><div class="note">需复核安全边界</div></div><div class="metric"><div class="ml">SHAP 已查看</div><div class="mv">6</div><div class="note">解释记录已同步</div></div></div><br>',unsafe_allow_html=True)
    l,r=st.columns([1.45,1])
    with l: st.markdown('<div class="sec"><div class="head"><div class="ic">▦</div><h3>最近患者列表</h3></div><table class="plist"><tr><th>匿名病例</th><th>阶段</th><th>最新监测</th><th>风险</th><th>状态</th></tr><tr><td>Case 014</td><td>Gn day 8</td><td>E2 1580 · max 18.0</td><td><span class="chip ct">低风险</span></td><td><span class="chip cp">待医生确认</span></td></tr><tr><td>Case 021</td><td>Gn day 6</td><td>E2 980 · max 14.5</td><td><span class="chip cw">中风险</span></td><td><span class="chip">已保存建议</span></td></tr><tr><td>Case 028</td><td>Gn day 10</td><td>E2 3120 · max 20.5</td><td><span class="chip cd">高风险</span></td><td><span class="chip cp">需复核</span></td></tr></table></div>',unsafe_allow_html=True)
    with r:
        st.markdown('<div class="sec"><div class="head"><div class="ic">＋</div><h3>快速入口</h3></div><div class="qcard"><div><b>新建患者记录</b><div class="note">录入基线与监测信息</div></div><span class="chip cp">患者录入</span></div><div class="qcard"><div><b>继续评估</b><div class="note">查看监测结果与下一次记录性预测</div></div><span class="chip ct">监测结果</span></div><div class="qcard"><div><b>解释复核</b><div class="note">查看三药 SHAP 贡献</div></div><span class="chip">SHAP</span></div></div>',unsafe_allow_html=True)
        if st.button("进入患者录入",type="primary",use_container_width=True): set_page("患者录入")
        if st.button("查看监测结果",use_container_width=True): set_page("监测结果")
    st.markdown('<div class="notice">AI 预测仅供临床辅助参考，最终解释权和用药决策权归主管医生。</div>',unsafe_allow_html=True)

def patient_page():
    st.markdown(PATIENT_CSS, unsafe_allow_html=True)
    title("患者临床信息录入","录入基线特征、历次监测信息、当前用药和卵泡评估，用于下一次记录性 Gn 剂量预测。")
    if st.session_state.validation_error: st.markdown(f'<div class="danger">{escape(st.session_state.validation_error)}</div>',unsafe_allow_html=True)
    v=st.session_state.patient.copy()
    records=[dict(r) for r in monitoring_records()]
    with st.form("patient_form"):
        head("▣","患者基础信息")
        c=st.columns(3)
        field_label(c[0],"年龄（岁）",True); v["age"]=c[0].number_input("年龄（岁）",18,55,int(v["age"]),label_visibility="collapsed",key="age")
        field_label(c[1],"BMI（kg/m²）",True); v["bmi"]=c[1].number_input("BMI（kg/m²）",14.0,40.0,float(v["bmi"]),step=.1,label_visibility="collapsed",key="bmi")
        field_label(c[2],"AMH（ng/mL）",True); v["amh"]=c[2].number_input("AMH（ng/mL）",0.0,20.0,float(v["amh"]),step=.1,label_visibility="collapsed",key="amh")
        c=st.columns(3)
        field_label(c[0],"AFC（个）",True); v["afc"]=c[0].number_input("AFC（个）",0,60,int(v["afc"]),label_visibility="collapsed",key="afc")
        field_label(c[1],"基础 FSH（IU/L）",True); v["basal_fsh"]=c[1].number_input("基础 FSH（IU/L）",0.0,40.0,float(v["basal_fsh"]),step=.1,label_visibility="collapsed",key="basal_fsh")
        field_label(c[2],"基础 LH（IU/L）",True); v["basal_lh"]=c[2].number_input("基础 LH（IU/L）",0.0,40.0,float(v["basal_lh"]),step=.1,label_visibility="collapsed",key="basal_lh")
        c=st.columns(3)
        field_label(c[0],"基础 E2（pg/mL）",True); v["basal_e2"]=c[0].number_input("基础 E2（pg/mL）",0.0,500.0,float(v["basal_e2"]),label_visibility="collapsed",key="basal_e2")
        field_label(c[1],"基础 P（ng/mL）",True); v["basal_p"]=c[1].number_input("基础 P（ng/mL）",0.0,10.0,float(v["basal_p"]),step=.1,label_visibility="collapsed",key="basal_p")
        field_label(c[2],"不孕年限（年）",False); v["years"]=c[2].number_input("不孕年限（年）",0,20,int(v["years"]),label_visibility="collapsed",key="years")
        c=st.columns(2)
        protocol_options=["GnRH-a 超长方案","GnRH-a 长方案","拮抗剂方案","其他"]
        protocol_value=str(v.get("protocol", protocol_options[0]))
        field_label(c[0],"促排方案",True); v["protocol"]=c[0].selectbox("促排方案",protocol_options,index=protocol_options.index(protocol_value) if protocol_value in protocol_options else 0,label_visibility="collapsed",key="protocol")
        infertility_options=["原发不孕","继发不孕"]
        infertility_value=str(v.get("infertility", infertility_options[0]))
        field_label(c[1],"不孕类型",False); v["infertility"]=c[1].selectbox("不孕类型",infertility_options,index=infertility_options.index(infertility_value) if infertility_value in infertility_options else 0,label_visibility="collapsed",key="infertility")
        head("▤","监测记录")
        for idx, rec in enumerate(records):
            rec["visit"] = idx + 1
            with st.container(border=True):
                st.markdown(f'<div class="monitor-record-title">\u7b2c {idx+1} \u6b21\u76d1\u6d4b\u8bb0\u5f55</div>', unsafe_allow_html=True)
                st.markdown('<div class="monitor-section-title">\u76d1\u6d4b\u65f6\u95f4</div>', unsafe_allow_html=True)
                c=st.columns(3)
                field_label(c[0],"\u4fc3\u6392\u5929\u6570 / stimulation day",True); rec["stim_day"]=c[0].number_input("\u4fc3\u6392\u5929\u6570 / stimulation day",1,30,int(rec.get("stim_day",idx+1)),label_visibility="collapsed",key=f"stim_day_{idx}")
                default_gap=0 if idx==0 else max(1, int(rec.get("stim_day", idx+1)) - int(records[idx-1].get("stim_day", idx)))
                field_label(c[1],"\u8ddd\u4e0a\u6b21\u590d\u8bca\u95f4\u9694\uff08\u5929\uff09",True); rec["days_since_previous_visit"]=c[1].number_input("\u8ddd\u4e0a\u6b21\u590d\u8bca\u95f4\u9694\uff08\u5929\uff09",0,30,int(rec.get("days_since_previous_visit",default_gap)),label_visibility="collapsed",key=f"days_since_previous_visit_{idx}")
                field_label(c[2],"\u76d1\u6d4b\u65e5\u671f",False); rec["monitoring_date"]=str(c[2].date_input("\u76d1\u6d4b\u65e5\u671f",parse_date(rec.get("monitoring_date",DEFAULT["monitoring_date"])),label_visibility="collapsed",key=f"monitoring_date_{idx}"))

                st.markdown('<div class="monitor-section-title">\u6fc0\u7d20\u548c\u5185\u819c\u4fe1\u606f</div>', unsafe_allow_html=True)
                c=st.columns(5)
                field_label(c[0],"\u5f53\u524d E2\uff08pg/mL\uff09",True); rec["e2"]=c[0].number_input("\u5f53\u524d E2\uff08pg/mL\uff09",0.0,10000.0,float(rec.get("e2",DEFAULT["e2"])),step=10.0,label_visibility="collapsed",key=f"e2_{idx}")
                field_label(c[1],"\u5f53\u524d LH\uff08IU/L\uff09",True); rec["lh_value"]=c[1].number_input("\u5f53\u524d LH\uff08IU/L\uff09",0.0,80.0,float(rec.get("lh_value",DEFAULT["lh_value"])),step=.1,label_visibility="collapsed",key=f"lh_value_{idx}")
                field_label(c[2],"\u5f53\u524d P\uff08ng/mL\uff09",True); rec["p"]=c[2].number_input("\u5f53\u524d P\uff08ng/mL\uff09",0.0,20.0,float(rec.get("p",DEFAULT["p"])),step=.1,label_visibility="collapsed",key=f"p_{idx}")
                field_label(c[3],"\u5f53\u524d FSH\uff08IU/L\uff09",True); rec["serum_fsh"]=c[3].number_input("\u5f53\u524d FSH\uff08IU/L\uff09",0.0,80.0,float(rec.get("serum_fsh",DEFAULT["serum_fsh"])),step=.1,label_visibility="collapsed",key=f"serum_fsh_{idx}")
                field_label(c[4],"\u5185\u819c\u539a\u5ea6\uff08mm\uff09",True); rec["current_endometrium"]=c[4].number_input("\u5185\u819c\u539a\u5ea6\uff08mm\uff09",0.0,30.0,float(rec.get("current_endometrium",DEFAULT["current_endometrium"])),step=.1,label_visibility="collapsed",key=f"endometrium_{idx}")

                st.markdown('<div class="monitor-section-title">\u5375\u6ce1\u4fe1\u606f</div>', unsafe_allow_html=True)
                c=st.columns(5)
                field_label(c[0],"\u603b\u5375\u6ce1\u6570\uff08\u4e2a\uff09",True); rec["total_follicles"]=c[0].number_input("\u603b\u5375\u6ce1\u6570\uff08\u4e2a\uff09",0,100,int(rec.get("total_follicles",DEFAULT["total_follicles"])),label_visibility="collapsed",key=f"total_follicles_{idx}")
                field_label(c[1],"\u5de6\u5375\u6ce1\u6570\uff08\u4e2a\uff09",False); rec["left_follicles"]=c[1].number_input("\u5de6\u5375\u6ce1\u6570\uff08\u4e2a\uff09",0,100,int(rec.get("left_follicles",DEFAULT["left_follicles"])),label_visibility="collapsed",key=f"left_follicles_{idx}")
                field_label(c[2],"\u53f3\u5375\u6ce1\u6570\uff08\u4e2a\uff09",False); rec["right_follicles"]=c[2].number_input("\u53f3\u5375\u6ce1\u6570\uff08\u4e2a\uff09",0,100,int(rec.get("right_follicles",DEFAULT["right_follicles"])),label_visibility="collapsed",key=f"right_follicles_{idx}")
                field_label(c[3],"\u6700\u5927\u5375\u6ce1\u5f84\uff08mm\uff09",True); rec["max_f"]=c[3].number_input("\u6700\u5927\u5375\u6ce1\u5f84\uff08mm\uff09",0.0,35.0,float(rec.get("max_f",DEFAULT["max_f"])),step=.5,label_visibility="collapsed",key=f"max_f_{idx}")
                field_label(c[4],"\u5e73\u5747\u5375\u6ce1\u5f84\uff08mm\uff09",True); rec["mean_f"]=c[4].number_input("\u5e73\u5747\u5375\u6ce1\u5f84\uff08mm\uff09",0.0,30.0,float(rec.get("mean_f",DEFAULT["mean_f"])),step=.1,label_visibility="collapsed",key=f"mean_f_{idx}")
                c=st.columns(5)
                field_label(c[0],"<10 mm \u5375\u6ce1\u6570",True); rec["f_lt10"]=c[0].number_input("<10 mm \u5375\u6ce1\u6570",0,80,int(rec.get("f_lt10",DEFAULT["f_lt10"])),label_visibility="collapsed",key=f"f_lt10_{idx}")
                field_label(c[1],"10-12 mm \u5375\u6ce1\u6570",True); rec["f_10_12"]=c[1].number_input("10-12 mm \u5375\u6ce1\u6570",0,80,int(rec.get("f_10_12",DEFAULT["f_10_12"])),label_visibility="collapsed",key=f"f_10_12_{idx}")
                field_label(c[2],"13-15 mm \u5375\u6ce1\u6570",True); rec["f_13_15"]=c[2].number_input("13-15 mm \u5375\u6ce1\u6570",0,80,int(rec.get("f_13_15",DEFAULT["f_13_15"])),label_visibility="collapsed",key=f"f_13_15_{idx}")
                field_label(c[3],"16-18 mm \u5375\u6ce1\u6570",True); rec["f_16_18"]=c[3].number_input("16-18 mm \u5375\u6ce1\u6570",0,80,int(rec.get("f_16_18",DEFAULT["f_16_18"])),label_visibility="collapsed",key=f"f_16_18_{idx}")
                field_label(c[4],"\u226518 mm \u5375\u6ce1\u6570",True); rec["f_gt18"]=c[4].number_input("\u226518 mm \u5375\u6ce1\u6570",0,80,int(rec.get("f_gt18",DEFAULT["f_gt18"])),label_visibility="collapsed",key=f"f_gt18_{idx}")
                bin_sum=sum(int(rec.get(key,0)) for key in FOLLICLE_BIN_KEYS)
                if int(rec.get("total_follicles",0)) != bin_sum:
                    st.markdown(f'<div class="mini-hint">\u603b\u5375\u6ce1\u6570 {int(rec.get("total_follicles",0))} \u4e0e\u5206\u5c42\u5408\u8ba1 {bin_sum} \u4e0d\u4e00\u81f4\uff0c\u5df2\u4fdd\u7559\u603b\u5375\u6ce1\u6570\u4f5c\u4e3a\u6a21\u578b\u8f93\u5165\u3002</div>', unsafe_allow_html=True)

                if idx < len(records) - 1:
                    st.markdown('<div class="monitor-section-title">\u5f53\u524d\u76d1\u6d4b\u6267\u884c\u7528\u836f\u8bb0\u5f55</div>', unsafe_allow_html=True)
                    source=str(rec.get("_executed_dose_default_source", ""))
                    hint="\u5df2\u6309\u4e0a\u4e00\u8f6e AI \u63a8\u8350\u9884\u586b\uff0c\u8bf7\u6309\u5b9e\u9645\u6267\u884c\u5242\u91cf\u6838\u5bf9\u3002" if source == "previous_ai_recommendation" else "\u8bf7\u586b\u5199\u8be5\u6b21\u76d1\u6d4b\u540e\u5b9e\u9645\u6267\u884c\u7684 FSH/LH/HMG \u5242\u91cf\uff0c\u7528\u4e8e\u4e0b\u4e00\u6b21\u8bb0\u5f55\u6027\u9884\u6d4b\u3002"
                    st.markdown(f'<div class="mini-hint">{hint}</div>', unsafe_allow_html=True)
                    c=st.columns(3)
                    field_label(c[0],"\u5f53\u524d\u76d1\u6d4b\u6267\u884c FSH \u5242\u91cf\uff08IU/\u5929\uff09",True); rec["current_fsh"]=c[0].number_input("\u5f53\u524d\u76d1\u6d4b\u6267\u884c FSH \u5242\u91cf\uff08IU/\u5929\uff09",0.0,600.0,float(rec.get("current_fsh",DEFAULT["current_fsh"])),step=25.0,label_visibility="collapsed",key=f"current_fsh_{idx}")
                    field_label(c[1],"\u5f53\u524d\u76d1\u6d4b\u6267\u884c LH \u5242\u91cf\uff08IU/\u5929\uff09",True); rec["current_lh"]=c[1].number_input("\u5f53\u524d\u76d1\u6d4b\u6267\u884c LH \u5242\u91cf\uff08IU/\u5929\uff09",0.0,300.0,float(rec.get("current_lh",DEFAULT["current_lh"])),step=37.5,label_visibility="collapsed",key=f"current_lh_{idx}")
                    field_label(c[2],"\u5f53\u524d\u76d1\u6d4b\u6267\u884c HMG \u5242\u91cf\uff08IU/\u5929\uff09",True); rec["current_hmg"]=c[2].number_input("\u5f53\u524d\u76d1\u6d4b\u6267\u884c HMG \u5242\u91cf\uff08IU/\u5929\uff09",0.0,450.0,float(rec.get("current_hmg",DEFAULT["current_hmg"])),step=37.5,label_visibility="collapsed",key=f"current_hmg_{idx}")
                else:
                    for key in DOSE_KEYS:
                        rec.pop(key, None)
                records[idx]=normalize_monitoring_record(rec)
        a,b,c,d=st.columns([1,1,1,1.25]); add=a.form_submit_button("+ 添加监测记录",use_container_width=True); save=b.form_submit_button("保存患者记录",use_container_width=True); reset=c.form_submit_button("清空重填",use_container_width=True); sub=d.form_submit_button("生成监测结果",type="primary",use_container_width=True)
    if add:
        records=mark_latest_record_as_executed(v, records)
        next_visit,next_stim,next_gap=next_monitoring_timing(records)
        new_record=default_monitoring_record(next_visit,next_stim,next_gap)
        for key in DOSE_KEYS:
            new_record.pop(key, None)
        records.append(new_record)
        persist_patient_records(v, records, recompute=False)
        rerun()
    if reset:
        persist_patient_records(DEFAULT.copy(), default_monitoring_records(), recompute=True); rerun()
    if save:
        persist_patient_records(v, records, recompute=True); st.success("\u60a3\u8005\u8bb0\u5f55\u5df2\u4fdd\u5b58\uff0c\u4ec5\u7528\u4e8e\u540e\u7eed\u8f85\u52a9\u51b3\u7b56\u3002")
    if sub:
        persist_patient_records(v, records, recompute=True)
        sync_recommendations(force=True, show_status=True, page_label="\u60a3\u8005\u5f55\u5165")
        set_page("\u76d1\u6d4b\u7ed3\u679c")


def _ohss_feature_value(item):
    feature=str(item.get("feature", "")); value=item.get("feature_value")
    try:
        n=float(value)
        if n != n: return "--"
        unit=OHSS_FEATURE_UNITS.get(feature, "")
        return f"{fmt(n)} {unit}".strip()
    except Exception:
        return str(value) if value not in (None, "") else "--"

def _ohss_feature_label(item):
    feature=str(item.get("feature", ""))
    return OHSS_FEATURE_LABELS.get(feature) or str(item.get("feature_cn") or feature or "--")

def ohss_breakdown_items(row):
    items=[x for x in (row.get("ohss_contributors") or []) if isinstance(x,Mapping)] if isinstance(row,Mapping) else []
    ordered=sorted(items,key=lambda x:abs(as_float(x.get("shap_value"),0)),reverse=True)
    rows=[]
    for item in ordered:
        feature=str(item.get("feature", ""))
        shap=as_float(item.get("shap_value"),0.0)
        rows.append({
            "feature": feature,
            "label": _ohss_feature_label(item),
            "value_label": _ohss_feature_value(item),
            "mean_abs_shap": abs(shap),
            "mean_shap": shap,
            "direction": "\u964d\u4f4e\u98ce\u9669" if shap < 0 else "\u589e\u52a0\u98ce\u9669",
            "source": "ohss_safety_warning_local_shap",
        })
    return rows

def ohss_change_text(row):
    current=next((x for x in st.session_state.recs if x.get("candidate_role")=="current"), None)
    base=ohss_display_profile(current or {}).get("display")
    prob=ohss_display_profile(row).get("display")
    if base is None or prob is None: return "--"
    delta=(prob-base)*100; sign="+" if delta>=0 else ""
    return f"{sign}{delta:.1f}% 显示"

def ohss_badge(row, include_prob=False):
    prof=ohss_profile(row); body=prof["category_zh"]
    if include_prob and prof["prob"] is not None: body=f'{ohss_display_pct(row)} \u00b7 {body}'
    return f'<span class="chip {prof["cls"]}">{escape(body)}</span>'

def visit_ohss_result(records_subset):
    if not (patient_form_to_snapshot is not None and records_subset and CANDIDATE_RESPONSE_AVAILABLE and score_candidate_response is not None):
        return None
    try:
        patient_ctx=patient_with_latest_monitoring(st.session_state.patient, records_subset)
        snap=patient_form_to_snapshot(patient_ctx)
        ref=reference_dose_context(patient_ctx, records_subset)
        result=score_candidate_response(
            snap,
            fsh_dose=ref["reference_fsh"],
            lh_dose=ref["reference_lh"],
            hmg_dose=ref["reference_hmg"],
            history_snapshots=temporal_history_snapshots(patient_ctx, records_subset),
        )
        prob=clamp(float(result.get("ohss_risk", result.get("ohss_risk_raw", 0.0))),0.0,1.0)
        source=str(result.get("ohss_run_id") or result.get("ohss_model_name") or result.get("run_id") or result.get("source") or "candidate_response_layer2_ohss")
        prof=ohss_profile(dict(ohss_risk_probability=prob,ohss_risk_threshold_low=OHSS_RISK_DEFAULTS["threshold_low"],ohss_risk_threshold_high=OHSS_RISK_DEFAULTS["threshold_high"]))
        row=dict(
            ohss_risk_probability=prob,
            safety_probability=prob,
            strict_ohss_probability=prob,
            ohss_source=source,
            ohss_contributors=[],
            ohss_risk_category=prof["category_en"],
            ohss_risk_percentile=prof["percentile"],
            ohss_risk_threshold_low=prof["threshold_low"],
            ohss_risk_threshold_high=prof["threshold_high"],
        )
        row.update(with_ohss_display_fields(row))
        return row
    except Exception as exc:
        st.session_state["visit_ohss_error"]=str(exc)
        return None


def follicle_load_text(record):
    total=fmt(derived_total_follicles(record)); mid=as_float(record.get("f_13_15",0),0)+as_float(record.get("f_16_18",0),0)+as_float(record.get("f_gt18",0),0); gt18=fmt(record.get("f_gt18"))
    return escape(f"\u603b {total} / \u226513mm {fmt(mid)} / >18mm {gt18}")

def ohss_warning_card(row):
    prof=ohss_profile(row)
    percentile="--" if prof["percentile"] is None else f'\u9ad8\u4e8e\u53c2\u8003\u4eba\u7fa4 {prof["percentile"]}%'
    st.markdown(f'''<div class="sec"><div class="ohss-card primary"><div class="ohss-title"><div><h3>\u4e2d\u91cd\u5ea6 OHSS \u65e9\u671f\u9884\u8b66</h3><div class="en">Moderate-to-severe OHSS early warning</div></div>{ohss_badge(row)}</div><div class="ohss-main"><div><div class="ohss-k">\u9884\u6d4b\u4e2d\u91cd\u5ea6 OHSS \u98ce\u9669</div><div class="ohss-v">{ohss_display_pct(row)}</div></div><div><div class="ohss-k">\u98ce\u9669\u5206\u7ea7</div><div class="ohss-v">{prof["category_zh"]}</div></div><div><div class="ohss-k">\u53c2\u8003\u4eba\u7fa4\u4f4d\u7f6e</div><div class="ohss-v" style="font-size:calc(22px * var(--fs-scale))">{escape(percentile)}</div></div></div><div class="ohss-sub">\u57fa\u4e8e\u5f53\u524d\u4fc3\u6392\u76d1\u6d4b\u4fe1\u606f\u53ca\u5386\u53f2\u7528\u836f\u8f68\u8ff9\u3002{OHSS_UI_DISCLAIMER}</div></div></div>''',unsafe_allow_html=True)


def ohss_shap_panel(row):
    items=[x for x in (row.get("ohss_contributors") or []) if isinstance(x,Mapping)]
    if not items:
        st.markdown('<div class="sec"><div class="head"><div class="ic">\u21af</div><h3>\u4e2a\u4f53\u5316\u98ce\u9669\u8d21\u732e\u56e0\u7d20</h3><span class="chip cm">Individual SHAP explanation</span></div><div class="warning">\u5f53\u524d OHSS \u5c40\u90e8 SHAP \u8d21\u732e\u6682\u672a\u8fd4\u56de\uff1b\u8bf7\u68c0\u67e5\u5b89\u5168\u9884\u8b66\u6a21\u578b\u670d\u52a1\u3002</div></div>',unsafe_allow_html=True); return
    pos=sorted([x for x in items if as_float(x.get("shap_value"),0)>0], key=lambda x:abs(as_float(x.get("shap_value"),0)), reverse=True)[:5]
    neg=sorted([x for x in items if as_float(x.get("shap_value"),0)<0], key=lambda x:abs(as_float(x.get("shap_value"),0)), reverse=True)[:5]
    max_abs=max([abs(as_float(x.get("shap_value"),0)) for x in pos+neg] or [1.0])
    def render_group(group, down=False):
        if not group: return '<div class="note">\u5f53\u524d\u672a\u8fd4\u56de\u8be5\u65b9\u5411\u7684\u4e3b\u8981\u8d21\u732e\u56e0\u7d20\u3002</div>'
        rows=[]
        for item in group:
            val=as_float(item.get("shap_value"),0); width=max(8,min(100,abs(val)/max_abs*100)); label=escape(_ohss_feature_label(item)); value=escape(_ohss_feature_value(item))
            chip='\u964d\u4f4e\u98ce\u9669' if down else '\u589e\u52a0\u98ce\u9669'; cls='down' if down else ''; ccls='ct' if down else 'cd'
            rows.append(f'<div class="risk-row"><div><div class="risk-name">{label}</div><div class="risk-val">{value}</div></div><div><div class="risk-track"><div class="risk-fill {cls}" style="width:{width:.0f}%"></div></div></div><div class="risk-meta"><span class="chip {ccls}">{chip}</span><span class="note">SHAP {val:+.3f}</span></div></div>')
        return ''.join(rows)
    html=f'''<div class="sec"><div class="head"><div class="ic">\u21af</div><h3>\u4e2a\u4f53\u5316\u98ce\u9669\u8d21\u732e\u56e0\u7d20</h3><span class="chip cm">Individual SHAP explanation</span></div><div class="ohss-factor-grid"><div class="ohss-card"><div class="ohss-factor-head">\u589e\u52a0\u98ce\u9669\u7684\u56e0\u7d20</div>{render_group(pos,False)}</div><div class="ohss-card"><div class="ohss-factor-head">\u964d\u4f4e\u98ce\u9669\u7684\u56e0\u7d20</div>{render_group(neg,True)}</div></div><div class="risk-note">SHAP values indicate model-attributed contribution and do not imply causality.</div></div>'''
    st.markdown(html,unsafe_allow_html=True)



def summary(b):
    fl=b["fsh_category"][0]
    st.markdown(
        f"""<div class="card pad summary-overview"><div class="head"><div class="ic">✦</div><h3>模型推荐总览</h3><span class="chip cp">获卵数 / OHSS 平衡参考</span></div><div class="sg" style="grid-template-columns:repeat(5,minmax(0,1fr))"><div class="tile dose"><div class="k">模型推荐 FSH</div><div class="v">{escape(fl)}</div></div><div class="tile dose"><div class="k">模型推荐 LH</div><div class="v">{fmt(b["lh"])} <span style="font-size:calc(13px * var(--fs-scale))">IU/天</span></div></div><div class="tile dose"><div class="k">模型推荐 HMG</div><div class="v">{fmt(b["hmg"])} <span style="font-size:calc(13px * var(--fs-scale))">IU/天</span></div></div><div class="tile"><div class="k">预测获卵数</div><div class="v">{fmt(b["o"])}</div></div><div class="tile"><div class="k">中重度 OHSS 风险</div><div class="v">{ohss_display_pct(b)}</div>{ohss_badge(b)}</div></div></div>""",
        unsafe_allow_html=True,
    )


def matrix(b):
    fl,fr=b["fsh_category"]; ll,lr=b["lh_category"]; hl,hr=b["hmg_category"]
    records=[dict(r) for r in monitoring_records()]
    if not records:
        st.markdown('<div class="sec"><div class="head"><div class="ic">\u2194</div><h3>\u6a2a\u5411\u52a8\u6001\u76d1\u6d4b\u89c6\u7a97</h3></div><div class="warning">\u5c1a\u672a\u6dfb\u52a0\u76d1\u6d4b\u8bb0\u5f55\u3002\u8bf7\u5148\u5728\u60a3\u8005\u5f55\u5165\u9875\u9762\u70b9\u51fb\u201c\u6dfb\u52a0\u76d1\u6d4b\u8bb0\u5f55\u201d\u3002</div></div>',unsafe_allow_html=True); return
    history=records[:-1]; today=records[-1]
    headers=['<th>\u65e5\u671f</th>']
    headers += [f'<th>\u7b2c {escape(fmt(r.get("stim_day")))} \u5929 / \u7b2c {escape(fmt(r.get("visit")))} \u6b21\u76d1\u6d4b</th>' for r in history]
    headers += [f'<th class="today">\u7b2c {escape(fmt(today.get("stim_day")))} \u5929\uff08\u4eca\u65e5\uff09</th><th class="pred">\u4e0b\u4e00\u6b21\u8bb0\u5f55\u6027\u9884\u6d4b</th>']
    def hist_values(key, formatter=lambda x: escape(fmt(x))): return [formatter(r.get(key)) for r in history]
    def row(label, cls, hist, today_value, pred_value):
        cells="".join(f"<td>{v}</td>" for v in hist)
        return f'<tr><td><span class="row-label {cls}">{escape(label)}</span></td>{cells}<td class="today">{today_value}</td><td class="pred">{pred_value}</td></tr>'
    rows=[row("FSH","row-fsh",hist_values("current_fsh",lambda x: escape(dose_class("fsh",x))),"-",pred_cell(fl)), row("LH","row-lh",hist_values("current_lh",lambda x: escape(dose_class("lh",x))),"-",pred_cell(ll)), row("HMG","row-hmg",hist_values("current_hmg",lambda x: escape(dose_class("hmg",x))),"-",pred_cell(hl)), row("\u6fc0\u7d20E2","row-e2",hist_values("e2"),escape(fmt(today.get("e2"))),"--"), row("\u6fc0\u7d20LH","row-lhv",hist_values("lh_value"),escape(fmt(today.get("lh_value"))),"--"), row("\u6fc0\u7d20P","row-p",hist_values("p"),escape(fmt(today.get("p"))),"--"), row("\u83b7\u5375\u6570","row-oocyte",["-" for _ in history],"-",pred_cell(fmt(b["o"]))), row("\u4e2d\u91cd\u5ea6 OHSS \u98ce\u9669","row-ohss",["" for _ in history],"",f'{ohss_badge(b, True)}<span class="pred-note">\u9884\u6d4b</span>')]
    st.markdown(f'<div class="sec"><div class="head"><div class="ic">\u2194</div><h3>\u6a2a\u5411\u52a8\u6001\u76d1\u6d4b\u89c6\u7a97</h3></div><div class="mw"><table class="matrix"><tr>{"".join(headers)}</tr>{"".join(rows)}</table></div></div>',unsafe_allow_html=True)


def cand_table():
    rows=[]
    candidates=sorted(
        [r for r in st.session_state.recs if r["candidate_role"]!="current"],
        key=lambda r:(
            as_float(r.get("balance_rank",999),999),
            as_float(r.get("candidate_total",9999),9999),
        ),
    )
    for i,x in enumerate(candidates[:8],1):
        plan=f'情景 {chr(64+i)}'
        role_chip='<span class="chip cp">模型推荐剂量</span>' if x.get("candidate_role")=="recommended" else ''
        fsh_range=display_dose_category("fsh",x["fsh"])
        rows.append(f'<tr><td><span class="rank">{i}</span></td><td>{escape(plan)} {role_chip}</td><td class="mono">{escape(fsh_range)}</td><td class="mono">{fmt(x["lh"])}</td><td class="mono">{fmt(x["hmg"])}</td><td class="mono">{fmt(x["o"])}</td><td class="mono">{ohss_display_pct(x)}</td></tr>')
    st.markdown(f'<div class="sec"><div class="head"><div class="ic">☷</div><h3>候选剂量情景分析</h3><span class="chip cp">获卵数 / OHSS 平衡排序</span></div><div class="mw"><table class="tbl"><tr><th>Rank</th><th>候选情景</th><th>FSH 范围</th><th>LH</th><th>HMG</th><th>预测获卵数</th><th>中重度 OHSS 风险</th></tr>{"".join(rows)}</table></div></div>',unsafe_allow_html=True)


def dose_model_notice():
    err=st.session_state.get("dose_recommendation_error")
    balance_err=st.session_state.get("candidate_balance_error")
    if err:
        st.markdown(f'<div class="warning">UI-reduced GRU(AddGate) \u5242\u91cf\u6a21\u578b\u6682\u672a\u8fd4\u56de\u6b63\u5f0f\u7ed3\u679c\uff1a{escape(str(err))}</div>', unsafe_allow_html=True)
    if balance_err:
        st.markdown(f'<div class="warning">获卵数与中重度 OHSS 候选平衡计算暂未完成：{escape(str(balance_err))}</div>', unsafe_allow_html=True)

def model_source_strip(page_label):
    ctx=st.session_state.get("dose_recommendation_context") or {}
    err=st.session_state.get("dose_recommendation_error")
    warnings=list(st.session_state.get("dose_recommendation_warnings") or [])
    if err:
        state="\u6a21\u578b\u7ed3\u679c\u9700\u590d\u6838"
        dot="err"
        state_chip='<span class="chip cw">fallback / review</span>'
    elif warnings:
        state="\u6a21\u578b\u5df2\u8fd4\u56de\uff0c\u5b58\u5728\u7279\u5f81\u63d0\u793a"
        dot="warn"
        state_chip=f'<span class="chip cw">{len(warnings)} \u6761\u6620\u5c04\u63d0\u793a</span>'
    elif ctx:
        state="\u6a21\u578b\u8ba1\u7b97\u5df2\u5b8c\u6210"
        dot=""
        state_chip='<span class="chip ct">\u56fa\u5b9a holdout \u6a21\u578b</span>'
    else:
        state="\u7b49\u5f85\u6a21\u578b\u8ba1\u7b97"
        dot="warn"
        state_chip='<span class="chip cm">pending</span>'
    deployment_mode=str(ctx.get("deployment_mode") or "holdout")
    protocol_label=(
        "\u56fa\u5b9a holdout \u7cbe\u7b80\u6a21\u578b"
        if deployment_mode == "holdout"
        else "5 \u6298 OOF \u5ba1\u8ba1\u5bf9\u7167"
    )
    task=escape(str(ctx.get("task") or "next-recorded absolute dose category prediction"))
    label=escape(str(page_label))
    st.markdown(
        f'<div class="model-strip"><div class="model-strip-left"><span class="model-dot {dot}"></span>'
        f'<div><div class="model-strip-title">{label} \u6a21\u578b\u72b6\u6001\uff1a{escape(state)}</div>'
        f'<div class="model-strip-sub">\u5242\u91cf\u6a21\u578b\uff1aUI-reduced GRU(AddGate) \u00b7 {escape(protocol_label)} \u00b7 {task}</div></div></div>'
        f'<div class="model-strip-tags">{state_chip}<span class="chip cp">分类模型学习医生剂量</span><span class="chip ct">V2 XGBoost 获卵数</span><span class="chip cw">\u4e2d\u91cd\u5ea6 OHSS \u98ce\u9669</span><span class="chip cp">获卵数 / OHSS Pareto 平衡</span></div></div>',
        unsafe_allow_html=True,
    )

def rec_page(monitor_only=False):
    title("\u52a8\u6001\u76d1\u6d4b\u4e0e\u7ed3\u679c\u9762\u677f" if monitor_only else "\u63a8\u8350\u65b9\u6848\u4e0e\u76d1\u6d4b\u7ed3\u679c","展示下一次记录性 Gn 剂量预测、预测获卵数、中重度 OHSS 风险和候选联合方案排序。")
    page_label="\u76d1\u6d4b\u7ed3\u679c" if monitor_only else "\u63a8\u8350\u65b9\u6848"
    refresh_page_recommendations(page_label)
    if not recommendation_required_notice(page_label):
        return
    b=best(auto_recompute=False); summary(b); dose_model_notice(); matrix(b)
    if not monitor_only:
        st.markdown('<div class="notice">模型推荐流程：剂量分类模型生成 FSH、LH、HMG 候选；V2 XGBoost 预测各候选情景的获卵数，中重度 OHSS 模型给出对应风险。系统先保留“获卵数不更低且 OHSS 风险不更高”的 Pareto 候选，再按候选集合内等权归一化距离选择最接近“较高获卵数、较低 OHSS 风险”理想点的平衡参考方案。KNN 仅用于相似历史病例校验。</div>',unsafe_allow_html=True)
        a,b1,c,d=st.columns([1.35,1,1,1])
        if a.button("\u4fdd\u5b58 AI \u5efa\u8bae\u4e0e\u533b\u751f\u786e\u8ba4",type="primary",use_container_width=True): st.success("\u5df2\u4fdd\u5b58 AI \u5efa\u8bae\u4e0e\u533b\u751f\u786e\u8ba4\u8bb0\u5f55\uff0c\u4ec5\u4f5c\u4e3a\u8f85\u52a9\u51b3\u7b56\u8bb0\u5f55\u3002")
        if b1.button("\u533b\u751f\u4fee\u6539\u5242\u91cf",use_container_width=True): st.session_state.show_modify=not st.session_state.get("show_modify",False)
        if c.button("\u67e5\u770b\u51b3\u7b56\u66f2\u7ebf",use_container_width=True): set_page("\u51b3\u7b56\u66f2\u7ebf")
        if d.button("\u67e5\u770b SHAP \u89e3\u91ca",use_container_width=True): set_page("\u63a8\u8350\u89e3\u91ca")
        if st.session_state.get("show_modify"):
            with st.container(border=True):
                bb=best(); st.markdown("#### \u533b\u751f\u4fee\u6539\u5242\u91cf")
                x,y,z=st.columns(3)
                ff=x.number_input("\u6700\u7ec8 FSH \u5242\u91cf",0.0,600.0,float(bb["fsh"]),step=25.0)
                ll=y.number_input("\u6700\u7ec8 LH \u5242\u91cf",0.0,300.0,float(bb["lh"]),step=37.5)
                hh=z.number_input("\u6700\u7ec8 HMG \u5242\u91cf",0.0,450.0,float(bb["hmg"]),step=37.5)
                ok=abs(ff-bb["fsh"])<1e-9 and abs(ll-bb["lh"])<1e-9 and abs(hh-bb["hmg"])<1e-9
                relation="\u4e00\u81f4" if ok else "\u5df2\u8c03\u6574"
                st.markdown(f'<div class="notice">\u533b\u751f\u6700\u7ec8\u5242\u91cf\u4e0e AI \u5efa\u8bae\u5173\u7cfb\uff1a<b>{relation}</b>\u3002\u4fdd\u5b58\u65f6\u8bb0\u5f55 AI recommended dose\u3001doctor final dose\u3001timestamp \u548c operator\u3002</div>',unsafe_allow_html=True)
    st.markdown('<div class="notice">AI \u4ec5\u63d0\u4f9b\u98ce\u9669\u5206\u5c42\u3001\u5019\u9009\u65b9\u6848\u6bd4\u8f83\u548c\u8f85\u52a9\u89e3\u91ca\uff1b\u5efa\u8bae\u7ed3\u5408\u4e34\u5e8a\u5224\u65ad\u3001E2\u3001\u5375\u6ce1\u8d1f\u8377\u548c\u60a3\u8005\u4e2a\u4f53\u60c5\u51b5\u8bc4\u4f30\u3002</div>',unsafe_allow_html=True)
def _scale(value, low, high, start, end):
    if high == low:
        return (start + end) / 2
    return start + (float(value) - low) / (high - low) * (end - start)

def _role_rank(role):
    return {"current":0,"recommended":1,"safe":2}.get(str(role),3)

def _axis_domain(values, min_span, floor=0.0, ceil=None):
    nums=[float(v) for v in values if v is not None]
    if not nums:
        nums=[0.0]
    low=min(nums); high=max(nums); span=high-low
    if span < min_span:
        center=(low+high)/2
        low=center-min_span/2
        high=center+min_span/2
    else:
        pad=span*.18
        low-=pad
        high+=pad
    if floor is not None:
        low=max(float(floor),low)
    if ceil is not None:
        high=min(float(ceil),high)
    if high <= low:
        high=low+float(min_span)
    return low, high

def _risk_axis_domain(values):
    nums=[max(0.0,min(100.0,float(v))) for v in values if v is not None]
    if not nums:
        nums=[0.0]
    low=math.floor(min(nums)*10.0+1e-9)/10.0
    high=math.ceil(max(nums)*10.0-1e-9)/10.0
    if high-low < 0.2-1e-9:
        high=low+0.2
    if high > 100.0:
        high=100.0
        low=max(0.0,high-0.2)
    return round(low,1),round(high,1)


def _recommended_curve_anchor():
    rows=[r for r in st.session_state.recs if isinstance(r,Mapping)]
    for row in rows:
        if row.get("candidate_role") == "recommended":
            return row
    candidates=[r for r in rows if r.get("candidate_role") != "current"]
    return max(candidates or rows, key=lambda r: as_float(r.get("score"), -1.0), default=None)


def _curve_delta_range(values):
    nums=[as_float(v,0.0) for v in values]
    if not nums:
        return "--"
    lo=min(nums); hi=max(nums)
    return f"{lo:+.1f}~{hi:+.1f}"


def _curve_metric_value(row, key, raw_counts=False):
    if key == "risk":
        return as_float(row.get("ohss_display_probability", row.get("safety_probability", row.get("ohss_risk_probability", 0.0))), 0.0) * 100
    if raw_counts and key == "o":
        return as_float(row.get(f"{key}_raw", row.get(key)), 0.0)
    return as_float(row.get(key), 0.0)

def _curve_value_text(value, key):
    if key == "risk":
        return f"{fmt(value, 2)}%"
    return f"{fmt(value, 1)} 个"


def _curve_delta_text(delta, key):
    if key == "risk":
        return f"{delta:+.2f} pp"
    return f"{delta:+.1f} 个"


def _curve_delta_class(delta, key):
    if abs(delta) < (0.005 if key == "risk" else 0.05):
        return "flat"
    if key == "risk":
        return "warn" if delta > 0 else "up"
    return "up" if delta > 0 else "down"


def _curve_sensitivity_html(points, dose_key, anchor):
    if not points or not isinstance(anchor, Mapping):
        return ""
    ordered = sorted(points, key=lambda r: as_float(r.get(dose_key), 0.0))
    low = ordered[0]
    high = ordered[-1]
    metrics = [("获卵数", "o"), ("OHSS", "risk")]
    rows = []
    for label, key in metrics:
        base_value = _curve_metric_value(anchor, key)
        cells = []
        for point, dose_label in ((low, "低剂量端"), (anchor, "推荐锚点"), (high, "高剂量端")):
            value = _curve_metric_value(point, key)
            delta = value - base_value
            delta_cls = _curve_delta_class(delta, key)
            cells.append(
                f'<div class="curve-s-cell"><span class="curve-s-dose">{dose_label} {fmt(point.get(dose_key))} IU</span>'
                f'<div class="curve-s-val">{_curve_value_text(value, key)}</div>'
                f'<span class="curve-s-delta {delta_cls}">{_curve_delta_text(delta, key)}</span></div>'
            )
        rows.append(f'<div class="curve-s-row"><div class="curve-s-label">{label}</div>{"".join(cells)}</div>')
    return '<div class="curve-sensitivity"><div class="curve-s-head"><span>条件预测对照</span><span>低剂量端</span><span>推荐锚点</span><span>高剂量端</span></div>' + ''.join(rows) + '</div>'


def _technical_sensitivity_panel():
    curve_cache=st.session_state.get("knn_curve_points", {})
    if not isinstance(curve_cache, Mapping):
        curve_cache={}
    base_row=_recommended_curve_anchor()
    drug_labels={"fsh":"FSH","lh":"LH","hmg":"HMG"}
    metric_defs=(("获卵数", "o"), ("OHSS", "risk"))
    cards=[]
    for drug,dose_key in (("fsh","fsh"),("lh","lh"),("hmg","hmg")):
        points=curve_cache.get(drug)
        if not points and base_row is not None:
            points=_local_curve_points(drug, base_row)
        if not points:
            continue
        points=sorted(points,key=lambda r:as_float(r.get(dose_key),0.0))
        anchor=next((r for r in points if r.get("candidate_role")=="recommended"), points[len(points)//2])
        endpoints=(("低剂量端",points[0]),("高剂量端",points[-1]))
        deltas=[]
        for _,point in endpoints:
            for _,metric_key in metric_defs:
                deltas.append(_curve_metric_value(point,metric_key,raw_counts=True)-_curve_metric_value(anchor,metric_key,raw_counts=True))
        max_abs=max([abs(x) for x in deltas] or [0.0])
        rows=[]
        for endpoint_label,point in endpoints:
            cells=[]
            for metric_label,metric_key in metric_defs:
                value=_curve_metric_value(point,metric_key,raw_counts=True)
                delta=value-_curve_metric_value(anchor,metric_key,raw_counts=True)
                width=6.0 if max_abs <= 1e-12 else max(8.0,min(100.0,abs(delta)/max_abs*100.0))
                cls=_curve_delta_class(delta,metric_key)
                cells.append(
                    f'<div class="tech-dose-metric"><div class="tech-dose-top"><span>{metric_label}</span><b>{_curve_delta_text(delta,metric_key)}</b></div>'
                    f'<div class="tech-dose-track"><div class="tech-dose-fill {cls}" style="width:{width:.1f}%"></div></div>'
                    f'<div class="tech-dose-value">{_curve_value_text(value,metric_key)}</div></div>'
                )
            rows.append(f'<div class="tech-dose-row"><div class="tech-dose-row-title">{endpoint_label} · {fmt(point.get(dose_key))} IU/天</div><div class="tech-dose-metrics">{"".join(cells)}</div></div>')
        cards.append(
            f'<div class="tech-dose-card"><div class="tech-dose-head"><div><div class="tech-dose-title">{drug_labels[drug]} 技术敏感性</div>'
            f'<div class="tech-dose-sub">相对推荐锚点 {fmt(anchor.get(dose_key))} IU/天</div></div><span class="chip cm">原始模型差值</span></div>{"".join(rows)}</div>'
        )
    if not cards:
        return ""
    return (
        '<div class="sec tech-sens"><div class="head"><div class="ic">∿</div><h3>技术敏感性参考</h3><span class="chip cw">非临床结论</span></div>'
        '<div class="notice">该区域仅用于检查候选反应模型对单药剂量扰动是否有方向性响应。条形长度为本卡片内归一化放大；获卵数显示 candidate-response 原始预测差值，OHSS 显示风险刻度差值；曲线与正式候选方案表使用同一严格中重度 OHSS 概率。主曲线直接显示真实条件预测，正式候选方案表使用相同模型输出。</div>'
        f'<div class="tech-dose-grid">{"".join(cards)}</div></div>'
    )


def _apply_knn_curve_sensitivity_gain(points, anchor):
    gain=max(1.0, as_float(KNN_CURVE_SENSITIVITY_GAIN, 1.0))
    if gain <= 1.0001 or not isinstance(anchor, Mapping):
        return points, gain
    base_o=as_float(anchor.get("o"),0.0)
    base_mii=as_float(anchor.get("mii"),0.0)
    calibrated=[]
    for item in points:
        row=dict(item)
        raw_o=as_float(row.get("o"),0.0)
        raw_mii=as_float(row.get("mii"),0.0)
        row["o_raw"]=round(raw_o,2)
        row["mii_raw"]=round(raw_mii,2)
        row["o"]=round(clamp(base_o + (raw_o-base_o)*gain,0,60),2)
        row["mii"]=round(min(clamp(base_mii + (raw_mii-base_mii)*gain,0,60), row["o"]),2)
        row["sensitivity_gain"]=gain
        calibrated.append(row)
    return calibrated, gain


def _curve_axis_ticks(drug):
    return {"fsh":[0,80,160,240], "lh":[0,75,150], "hmg":[0,75,150,225]}[drug]


def _curve_display_levels(drug):
    return {"fsh":[40,120,200], "lh":[0,75,150], "hmg":[0,75,150,225]}[drug]


def _curve_display_dose(drug, dose):
    dose=as_float(dose,0.0)
    if drug == "fsh":
        if dose < 80:
            return 40.0
        if dose <= 160:
            return 120.0
        return 200.0
    if drug == "lh":
        if dose <= 0:
            return 0.0
        if dose < 150:
            return 75.0
        return 150.0
    if dose <= 0:
        return 0.0
    if dose < 150:
        return 75.0
    if dose < 225:
        return 150.0
    return 225.0


def _local_curve_points(drug, base_row):
    if not isinstance(base_row, Mapping):
        return []
    dose_key={"fsh":"fsh","lh":"lh","hmg":"hmg"}[drug]
    base_doses={"fsh":as_float(base_row.get("fsh"),0.0),"lh":as_float(base_row.get("lh"),0.0),"hmg":as_float(base_row.get("hmg"),0.0)}
    center_actual=base_doses[dose_key]
    center=_curve_display_dose(drug, center_actual)
    dose_values=sorted({float(x) for x in _curve_display_levels(drug) + [center]})
    records=monitoring_records()
    patient_for_curve=patient_with_latest_monitoring(st.session_state.get("patient", {}), records)
    snap=None
    if patient_form_to_snapshot is not None:
        try:
            snap=patient_form_to_snapshot(patient_for_curve)
        except Exception as exc:
            st.session_state["dose_curve_snapshot_error"]=str(exc)
    history_snapshots=temporal_history_snapshots(patient_for_curve, records)
    points=[]
    for value in dose_values:
        combo=dict(base_doses)
        combo[dose_key]=value
        row=dict(base_row)
        row.update(combo)
        row["candidate_role"]="recommended" if abs(value-center)<1e-9 else "local"
        row["name"]="recommended anchor" if row["candidate_role"]=="recommended" else "local perturbation"
        row["dose_range_label"]=display_dose_category(drug, center_actual if row["candidate_role"]=="recommended" else value)
        row["dose_axis_value"]=value
        if row["candidate_role"] != "recommended" and snap is not None and CANDIDATE_RESPONSE_AVAILABLE and score_candidate_response is not None:
            try:
                real=score_candidate_response(
                    snap,
                    fsh_dose=combo["fsh"],
                    lh_dose=combo["lh"],
                    hmg_dose=combo["hmg"],
                    bound_counts=True,
                    history_snapshots=history_snapshots,
                )
                o=clamp(float(real["oocytes"]),0,60)
                prob=clamp(float(real.get("ohss_risk", real.get("ohss_risk_raw", 0.0))),0,1)
                row["count_bound_applied"]=bool(real.get("count_bound_applied", True))
                row["o"]=round(o,2)
                row["o_selection_value"]=float(o)
                row["strict_ohss_probability"]=prob
                row["safety_probability"]=prob
                row["ohss_risk_probability"]=prob
                row["ohss_source"]=str(real.get("ohss_run_id") or real.get("ohss_model_name") or real.get("run_id") or real.get("source") or "candidate_response_layer2_ohss")
                row["response_source"]=str(real.get("source") or real.get("model_name") or "candidate_response")
            except Exception as exc:
                st.session_state["dose_curve_candidate_response_error"]=str(exc)
        else:
            prob=_as_prob(row.get("ohss_risk_probability", row.get("strict_ohss_probability", row.get("safety_probability"))))
            if prob is not None:
                row["strict_ohss_probability"]=prob
                row["safety_probability"]=prob
                row["ohss_risk_probability"]=prob
        for stale in ("ohss_gate_threshold","ohss_gate_pass","ohss_gate_status"):
            row.pop(stale,None)
        row["curve_recommended"]=row["candidate_role"]=="recommended"
        row["curve_recommendation_basis"]="formal oocyte/OHSS balanced recommendation"
        row.update(with_ohss_display_fields(row))
        points.append(row)
    return sorted(points,key=lambda r:as_float(r.get(dose_key),0.0))


def dose_curve(drug, title_, color):
    dose_key={"fsh":"fsh","lh":"lh","hmg":"hmg"}[drug]
    drug_label={"fsh":"FSH","lh":"LH","hmg":"HMG"}[drug]
    axis_ticks=_curve_axis_ticks(drug)
    base_row=_recommended_curve_anchor()
    points=_local_curve_points(drug, base_row)
    if not points:
        points=[r for r in st.session_state.recs if isinstance(r,Mapping)]
        points=sorted(points,key=lambda r:(as_float(r.get(dose_key),0),_role_rank(r.get("candidate_role")),str(r.get("name",""))))
    if not points:
        return f'<div class="svgcard"><h4>{escape(title_)}</h4><div class="warning">暂无候选方案数据。</div></div>'
    anchor=next((r for r in points if r.get("curve_recommended")),
                next((r for r in points if r.get("candidate_role")=="recommended"), points[len(points)//2]))
    render_points=list(points)
    if drug=="fsh":
        if as_float(render_points[0].get(dose_key),min(axis_ticks))>min(axis_ticks):
            left=dict(render_points[0]); left[dose_key]=float(min(axis_ticks)); render_points.insert(0,left)
        if as_float(render_points[-1].get(dose_key),max(axis_ticks))<max(axis_ticks):
            right=dict(render_points[-1]); right[dose_key]=float(max(axis_ticks)); render_points.append(right)
    try:
        st.session_state.setdefault("knn_curve_points", {})[drug]=points
        st.session_state.setdefault("knn_curve_axis_ticks", {})[drug]=axis_ticks
    except Exception:
        pass
    x_min=float(min(axis_ticks)); x_max=float(max(axis_ticks))
    count_values=[as_float(r.get("o"),0.0) for r in points]
    def risk_prob(row):
        return as_float(row.get("ohss_risk_probability",row.get("strict_ohss_probability",row.get("safety_probability",0.0))),0.0)
    risk_values=[risk_prob(r)*100 for r in points]
    count_low,count_high=_axis_domain(count_values,0.18,0.0)
    risk_low,risk_high=_risk_axis_domain(risk_values)
    try:
        st.session_state.setdefault("knn_curve_risk_domains", {})[drug]=(risk_low,risk_high)
    except Exception:
        pass
    base_o=as_float(anchor.get("o"),0.0)
    base_risk=risk_prob(anchor)*100
    base_o_raw=as_float(anchor.get("o_raw",anchor.get("o")),0.0)
    for r in points:
        r["_o_delta"]=as_float(r.get("o"),0.0)-base_o
        r["_o_raw_delta"]=as_float(r.get("o_raw",r.get("o")),0.0)-base_o_raw
        r["_risk_delta"]=risk_prob(r)*100-base_risk
    x0,y0,w,h=58,204,308,166
    def sx(x): return _scale(x,x_min,x_max,x0,x0+w)
    def sy_count(y): return _scale(y,count_low,count_high,y0,y0-h)
    def sy_risk(y): return _scale(float(y)*100,risk_low,risk_high,y0,y0-h)
    def pline(value_key, yfunc):
        return " ".join(f'{sx(as_float(r.get(dose_key),0)):.1f},{yfunc(as_float(r.get(value_key),0)):.1f}' for r in render_points)
    oocyte_points=pline("o",sy_count)
    risk_points=" ".join(f'{sx(as_float(r.get(dose_key),0)):.1f},{sy_risk(risk_prob(r)):.1f}' for r in render_points)
    y_ticks=[count_low,(count_low+count_high)/2,count_high]
    risk_ticks=[risk_low,(risk_low+risk_high)/2,risk_high]
    count_tick_digits=2 if count_high-count_low < 0.25 else 1
    grid="".join(f'<line x1="{x0}" y1="{sy_count(v):.1f}" x2="{x0+w}" y2="{sy_count(v):.1f}"/><text x="45" y="{sy_count(v)+4:.1f}" text-anchor="end">{fmt(v,count_tick_digits)}</text>' for v in y_ticks)
    rticks="".join(f'<text x="374" y="{_scale(v,risk_low,risk_high,y0,y0-h)+4:.1f}">{fmt(v,1)}%</text>' for v in risk_ticks)
    xticks="".join(f'<line x1="{sx(v):.1f}" y1="{y0}" x2="{sx(v):.1f}" y2="{y0+5}"/><text x="{sx(v):.1f}" y="{y0+20}" text-anchor="middle">{fmt(v)}</text>' for v in axis_ticks)
    markers=[]
    for r in points:
        if bool(r.get("curve_recommended")):
            x=sx(as_float(r.get(dose_key),0))
            yo=sy_count(as_float(r.get("o"),0))
            yr=sy_risk(risk_prob(r))
            label=escape(display_dose_category(drug, r.get("dose_range_label", r.get(dose_key))))
            point_o=as_float(r.get("o"),0.0)
            point_risk=risk_prob(r)*100
            markers.append(
                f'<g data-recommended-point="true" data-recommended-oocytes="{point_o:.2f}" data-recommended-ohss="{point_risk:.4f}">'
                f'<line data-recommended-marker="vertical-dashed" x1="{x:.1f}" y1="{y0-h}" x2="{x:.1f}" y2="{y0}" stroke="#3525cd" stroke-width="1.6" stroke-dasharray="5 5" opacity=".75"/>'
                f'<circle cx="{x:.1f}" cy="{yo:.1f}" r="5" fill="#14b8a6" stroke="#fff" stroke-width="2.5"/>'
                f'<circle cx="{x:.1f}" cy="{yr:.1f}" r="5" fill="#f59e0b" stroke="#fff" stroke-width="2.5"/>'
                f'</g>'
            )
    axis_attr=','.join(str(int(x)) for x in axis_ticks)
    anchor_label=escape(display_dose_category(drug, anchor.get("dose_range_label", anchor.get(dose_key))))
    anchor_o=as_float(anchor.get("o"),0.0)
    anchor_risk=risk_prob(anchor)*100
    return f"""<div class="svgcard" data-axis-ticks="{axis_attr}"><div style="display:flex;align-items:center;justify-content:space-between;gap:10px"><h4 style="margin:0;font-size:calc(20px * var(--fs-scale));line-height:1.25;white-space:nowrap">{escape(title_)}</h4><span class="chip cp">获卵数 / OHSS 平衡</span></div><div class="curve-rec-summary"><span class="dose">模型推荐 {anchor_label}</span><span>获卵数 <b>{fmt(anchor_o)}</b></span><span>OHSS <b>{fmt(anchor_risk,2)}%</b></span></div><svg viewBox="0 0 420 278" width="100%" role="img" aria-label="{escape(title_)}"><g stroke="#dbe3f0" stroke-width="1">{grid}<line x1="{x0}" y1="{y0-h}" x2="{x0}" y2="{y0}"/><line x1="{x0}" y1="{y0}" x2="{x0+w}" y2="{y0}"/><line x1="{x0+w}" y1="{y0-h}" x2="{x0+w}" y2="{y0}"/></g><g fill="#64748b" font-size="10">{rticks}{xticks}<text x="{x0+w/2}" y="264" text-anchor="middle">{drug_label} 剂量（IU/天）</text><text x="15" y="106" transform="rotate(-90 15,106)" text-anchor="middle">预测获卵数（个）</text><text x="407" y="106" transform="rotate(90 407,106)" text-anchor="middle">中重度 OHSS 风险（%）</text></g><polyline points="{oocyte_points}" fill="none" stroke="#14b8a6" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/><polyline points="{risk_points}" fill="none" stroke="#f59e0b" stroke-width="3.2" stroke-dasharray="7 5" stroke-linecap="round" stroke-linejoin="round"/>{''.join(markers)}<g font-size="11" font-weight="800"><circle cx="216" cy="22" r="4" fill="#14b8a6"/><text x="224" y="26" fill="#14b8a6">获卵数</text><line x1="276" y1="22" x2="294" y2="22" stroke="#f59e0b" stroke-width="3.2" stroke-dasharray="7 5"/><text x="298" y="26" fill="#d97706">中重度 OHSS 风险</text></g></svg></div>"""


def knn_evidence_panel():
    ctx=current_layer1_context()
    knn_groups={drug:[] for drug in ("fsh","lh","hmg")}
    if UI_REAL_DATA_AVAILABLE and build_knn_drug_summary is not None:
        try:
            knn_groups=build_knn_drug_summary(ctx,limit=3)
            st.session_state.pop("knn_ui_error", None)
        except Exception as exc:
            st.session_state["knn_ui_error"]=str(exc)
    if any(knn_groups.get(drug) for drug in ("fsh","lh","hmg")):
        st.session_state["knn_drug_summary"]=knn_groups
        cards=[]
        for drug,label in (("fsh","FSH"),("lh","LH"),("hmg","HMG")):
            rows=knn_groups.get(drug) or []
            body="".join(
                f'<tr><td>{escape(r["case"])}</td><td class="mono">{escape(r["dose"])}</td><td>{int(r["case_count"])}</td><td>{escape(r["selection_rate"])}</td><td>{escape(r["success_rate"])}</td></tr>'
                for r in rows
            )
            cards.append(
                f'<div class="knn-drug-card"><div class="knn-drug-title"><span>{label} 相似病例</span></div><table class="knn-mini-table"><tr><th>病例</th><th>剂量</th><th>病例数</th><th>选择率</th><th>成功率</th></tr>{body}</table></div>'
            )
        st.markdown(
            f'<div class="sec"><div class="knn-evidence-head"><div class="ic">≋</div><h3>KNN 相似历史病例支持</h3><span class="chip cm">推荐解释与历史校验</span></div><div class="knn-drug-grid">{"".join(cards)}</div><div class="knn-evidence-note">各药物按选择率与成功率等权综合排序；仅用于解释和历史证据校验，不参与推荐剂量选择。</div></div>',
            unsafe_allow_html=True,
        )
    else:
        detail=escape(st.session_state.get("knn_ui_error") or st.session_state.get("layer1_error") or "未读取到实时 KNN 相似历史病例。")
        st.markdown(f'<div class="warning">KNN 相似历史病例暂不可用：{detail}</div>',unsafe_allow_html=True)


def knn_page():
    title("决策曲线","比较候选剂量下的预测获卵数与中重度 OHSS 风险，并查看联合候选剂量情景。")
    refresh_page_recommendations("决策曲线")
    if not recommendation_required_notice("决策曲线"):
        return
    st.session_state["knn_curve_points"]={}
    st.session_state["knn_curve_axis_ticks"]={}
    st.session_state["knn_curve_risk_domains"]={}
    st.markdown('<div class="notice">固定其他患者变量，仅改变单种 Gn 剂量进行条件情景分析；虚线和实心点标出正式模型推荐剂量及其预测结果。</div>',unsafe_allow_html=True)
    st.markdown('<div class="sec"><div class="head"><div class="ic">∿</div><h3>剂量-反应曲线</h3></div><div class="curves">'+dose_curve("fsh","FSH","#4f46e5")+dose_curve("lh","LH","#0f766e")+dose_curve("hmg","HMG","#4f46e5")+'</div></div>',unsafe_allow_html=True)
    cand_table()

def factor(items):
    rows=[]
    for item in items:
        if isinstance(item,Mapping):
            n=item.get("label","--"); t=item.get("direction","--"); w=item.get("width",40); c=item.get("fill",""); chip=item.get("chip","cp")
        else:
            n,t,w,c,chip=item
        rows.append(f'<div class="factor"><div>{escape(str(n))}</div><div class="bar"><div class="fill {escape(str(c))}" style="width:{int(w)}%"></div></div><div class="fc"><span class="chip {escape(str(chip))}">{escape(str(t))}</span></div></div>')
    return "".join(rows)

def feature_value_text(feature, patient):
    key=str(feature)
    p=patient
    values={
        "Day": p.get("stim_day"),
        "gn_day": p.get("stim_day"),
        "cycle_day": p.get("stim_day"),
        "days_since_previous_visit": p.get("days_since_previous_visit"),
        "age": p.get("age"),
        "bmi": p.get("bmi"),
        "infertility_duration": p.get("years"),
        "amh": p.get("amh"),
        "afc": p.get("afc"),
        "initial_gn_dose": p.get("initial_gn"),
        "basal_fsh": p.get("basal_fsh"),
        "basal_lh": p.get("basal_lh"),
        "basal_e2": p.get("basal_e2"),
        "basal_p": p.get("basal_p"),
        "current_e2": p.get("e2"),
        "current_lh": p.get("lh_value"),
        "current_p": p.get("p"),
        "current_fsh": p.get("serum_fsh"),
        "current_endometrium": p.get("current_endometrium"),
        "previous_fsh_daily_dose": p.get("reference_fsh"),
        "previous_lh_daily_dose": p.get("reference_lh"),
        "previous_hmg_daily_dose": p.get("reference_hmg"),
        "previous_lh_like_hmg_daily_dose": as_float(p.get("reference_lh"),0)+as_float(p.get("reference_hmg"),0),
        "previous_gn_dose": p.get("previous_gn_dose"),
        "mean_follicle_diameter": p.get("mean_f"),
        "max_follicle_diameter": p.get("max_f"),
        "monitoring_order": p.get("visit"),
        "visits_seen": p.get("visit"),
        "total_follicle_count": derived_total_follicles(p),
        "left_follicle_count": p.get("left_follicles"),
        "right_follicle_count": p.get("right_follicles"),
        "follicle_count_lt_10": p.get("f_lt10"),
        "follicle_count_10_12": p.get("f_10_12"),
        "follicle_count_13_15": p.get("f_13_15"),
        "follicle_count_16_18": p.get("f_16_18"),
        "follicle_count_gt_18": p.get("f_gt18"),
    }
    value=values.get(key)
    if value is None:
        return "--"
    return fmt(value)

def shap_lookup(items):
    lookup={}
    for item in items:
        if not isinstance(item, Mapping):
            continue
        feature=str(item.get("feature","")).strip()
        if feature:
            lookup[feature]=item
    return lookup

def patient_value(patient, key):
    if key == "total_follicle_count":
        return derived_total_follicles(patient)
    if key == "previous_lh_like_hmg_daily_dose":
        return as_float(patient.get("reference_lh"),0)+as_float(patient.get("reference_hmg"),0)
    if key == "previous_gn_dose":
        return as_float(patient.get("reference_fsh"),0)+as_float(patient.get("reference_lh"),0)+as_float(patient.get("reference_hmg"),0)
    mapping={
        "Day":"stim_day",
        "gn_day":"stim_day",
        "evaluation_day":"stim_day",
        "cycle_day":"stim_day",
        "current_e2":"e2",
        "current_lh":"lh_value",
        "current_p":"p",
        "current_fsh":"serum_fsh",
        "current_endometrium":"current_endometrium",
        "total_follicle_count":"total_follicles",
        "max_follicle_diameter":"max_f",
        "mean_follicle_diameter":"mean_f",
        "follicle_count_lt_10":"f_lt10",
        "follicle_count_10_12":"f_10_12",
        "follicle_count_13_15":"f_13_15",
        "follicle_count_16_18":"f_16_18",
        "follicle_count_gt_18":"f_gt18",
        "monitoring_order":"visit",
        "visits_seen":"visit",
        "previous_fsh_daily_dose":"reference_fsh",
        "previous_lh_daily_dose":"reference_lh",
        "previous_hmg_daily_dose":"reference_hmg",
        "infertility_duration":"years",
        "male_age":"male_age",
    }
    return patient.get(mapping.get(str(key), str(key)))

def patient_breakdown_items(drug, shap_items, patient):
    lookup=shap_lookup(shap_items)
    fallback_mean={
        "fsh":{"current_e2":.028,"total_follicle_count":.021,"follicle_count_gt_18":-.018,"previous_fsh_daily_dose":.016},
        "lh":{"current_lh":-.024,"current_p":-.015,"current_e2":.014,"previous_lh_daily_dose":.012},
        "hmg":{"previous_hmg_daily_dose":.022,"mean_follicle_diameter":-.018,"follicle_count_16_18":-.015,"current_e2":-.013},
    }.get(drug,{})
    common=[
        ("Day","促排天数",.020),
        ("current_e2","E2(pg/mL)",.030),
        ("current_lh","血清LH(IU/L)",.022),
        ("current_p","P(ng/mL)",.018),
        ("current_fsh","血清FSH(IU/L)",.018),
        ("current_endometrium","内膜(mm)",.017),
        ("total_follicle_count","总卵泡数",.028),
        ("max_follicle_diameter","最大卵泡(mm)",.026),
        ("mean_follicle_diameter","平均卵泡(mm)",.024),
        ("follicle_count_16_18","16-18mm卵泡",.020),
        ("follicle_count_gt_18","≥18mm卵泡",.020),
        ("amh","AMH",.022),
        ("afc","AFC",.022),
        ("age","年龄",.018),
        ("bmi","BMI",.018),
        ("basal_fsh","基础FSH",.014),
        ("basal_lh","基础LH",.014),
        ("basal_e2","基础E2",.012),
    ]
    drug_specific={
        "fsh":[
            ("previous_fsh_daily_dose","历史FSH(IU/天)",.026),
            ("current_e2","E2(pg/mL)",.030),
            ("total_follicle_count","总卵泡数",.028),
            ("max_follicle_diameter","最大卵泡(mm)",.026),
            ("mean_follicle_diameter","平均卵泡(mm)",.024),
            ("follicle_count_16_18","16-18mm卵泡",.020),
            ("follicle_count_gt_18","≥18mm卵泡",.020),
            ("amh","AMH",.022),
            ("afc","AFC",.022),
            ("basal_fsh","基础FSH",.014),
        ],
        "lh":[
            ("previous_lh_daily_dose","历史LH(IU/天)",.026),
            ("current_lh","血清LH(IU/L)",.028),
            ("current_p","P(ng/mL)",.020),
            ("current_e2","E2(pg/mL)",.024),
            ("Day","促排天数",.020),
            ("current_endometrium","内膜(mm)",.017),
            ("total_follicle_count","总卵泡数",.022),
            ("basal_lh","基础LH",.014),
            ("amh","AMH",.020),
            ("afc","AFC",.020),
        ],
        "hmg":[
            ("previous_hmg_daily_dose","历史HMG(IU/天)",.030),
            ("previous_lh_like_hmg_daily_dose","历史LH+HMG",.026),
            ("mean_follicle_diameter","平均卵泡(mm)",.026),
            ("max_follicle_diameter","最大卵泡(mm)",.024),
            ("follicle_count_13_15","13-15mm卵泡",.021),
            ("follicle_count_16_18","16-18mm卵泡",.020),
            ("follicle_count_gt_18","≥18mm卵泡",.020),
            ("current_e2","E2(pg/mL)",.023),
            ("total_follicle_count","总卵泡数",.022),
            ("bmi","BMI",.018),
        ],
    }
    ordered=drug_specific.get(drug, common)
    rows=[]
    seen=set()
    for feature,label,fallback_abs in ordered + common:
        if feature in seen:
            continue
        seen.add(feature)
        value=patient_value(patient, feature)
        if value is None:
            continue
        src=lookup.get(feature,{})
        mean_abs=float(src.get("mean_abs_shap",fallback_abs))
        mean=float(src.get("mean_shap",fallback_mean.get(feature, 0.004 if mean_abs >= 0 else -0.004)))
        rows.append({
            "feature":feature,
            "label":label,
            "value_label":fmt(value),
            "mean_abs_shap":mean_abs,
            "mean_shap":mean,
        })
        if len(rows) >= 10:
            break
    return rows

def plus_num(x):
    return f"{float(x):+.3f}"

def ui_input_contribution_items(items):
    return [
        item for item in items
        if isinstance(item,Mapping) and str(item.get("feature","")) in UI_INPUT_ATTRIBUTION_FEATURES
    ]

def candidate_shap_breakdown_items(payload, patient, probability=False):
    source=str(payload.get("method","candidate_response_tree_shap")) if isinstance(payload,Mapping) else "candidate_response_tree_shap"
    scale=100.0 if probability else 1.0
    rows=[]
    for item in (payload.get("items") or []) if isinstance(payload,Mapping) else []:
        if not isinstance(item,Mapping):
            continue
        feature=str(item.get("feature",""))
        value=as_float(item.get("mean_shap"),0.0)*scale
        clinical=feature in UI_INPUT_ATTRIBUTION_FEATURES
        rows.append({
            "feature":feature,
            "label":OHSS_FEATURE_LABELS.get(feature,feature),
            "value_label":feature_value_text(feature,patient) if clinical else "--",
            "mean_abs_shap":abs(value),
            "mean_shap":value,
            "source":source,
            "clinical_display":clinical,
        })
    rows.sort(key=lambda item:abs(as_float(item.get("mean_shap"),0.0)),reverse=True)
    return rows

def breakdown_card(
    drug,
    prediction_label,
    probability,
    items,
    patient,
    badge_label=None,
    baseline_probability=None,
    baseline_display=None,
    all_shap_abs_sum=None,
    other_items=None,
    show_other=False,
    contribution_unit="",
):
    available=[item for item in items if isinstance(item,Mapping)]
    is_ohss_available=any(
        "layer2_ohss" in str(item.get("source",""))
        or str(item.get("source","")).startswith("ohss_safety_warning_local")
        for item in available
    )
    if is_ohss_available:
        lowering=[item for item in available if as_float(item.get("mean_shap"),0.0)<0][:3]
        increasing=[item for item in available if as_float(item.get("mean_shap"),0.0)>0][:3]
        selected=lowering+increasing
        selected_ids={id(item) for item in selected}
        selected.extend(
            item for item in available
            if id(item) not in selected_ids and len(selected)<MAIN_LOCAL_SHAP_LIMIT
        )
        selected=selected[:MAIN_LOCAL_SHAP_LIMIT]
        selected_ids={id(item) for item in selected}
        unshown=[item for item in available if id(item) not in selected_ids]
    else:
        selected=available[:MAIN_LOCAL_SHAP_LIMIT]
        unshown=available[MAIN_LOCAL_SHAP_LIMIT:]
    detail_items=[item for item in (other_items if other_items is not None else unshown) if isinstance(item,Mapping)]
    display_scores=[float(item.get("mean_shap",0.0)) for item in selected]
    other_value=sum(as_float(item.get("mean_shap"),0.0) for item in detail_items)
    include_other=bool(show_other) and bool(detail_items)
    max_abs=max([abs(value) for value in display_scores]+([abs(other_value)] if include_other else []) or [1.0])
    is_ohss_local=is_ohss_available
    is_candidate_local=any(str(item.get("source","")).startswith("candidate_response_") for item in selected)
    is_local=is_ohss_local or is_candidate_local or any(
        str(item.get("source","")).startswith(("phase867_oof_local","phase870_current_patient"))
        for item in selected
    )
    rows=[]
    for item,display_score in zip(selected,display_scores):
        feature=str(item.get("feature",item.get("label","--")))
        label=str(item.get("label",feature))
        value=str(item.get("value_label") or feature_value_text(feature,patient))
        mean=float(item.get("mean_shap",0.0))
        mag=abs(float(item.get("mean_abs_shap",abs(display_score))))
        width=int(round(28+64*(mag/max_abs if max_abs else 0)))
        width=max(18,min(96,width))
        neg=mean<0
        if is_ohss_local:
            risk_up=mean>0
            chip="\u589e\u52a0\u98ce\u9669" if risk_up else "\u964d\u4f4e\u98ce\u9669"
        elif is_candidate_local:
            risk_up=neg
            chip="\u964d\u4f4e\u9884\u6d4b" if neg else "\u589e\u52a0\u9884\u6d4b"
        else:
            risk_up=neg
            chip=str(item.get("direction") or ("\u8d1f\u5411\u8d21\u732e" if neg else "\u6b63\u5411\u8d21\u732e"))
            chip=chip.replace("\u5e73\u5747", "\u5c40\u90e8") if is_local else chip
        fill_cls="bd-fill neg" if risk_up else "bd-fill"
        chip_cls="bd-chip neg" if risk_up else "bd-chip"
        score_html=f'<div class="bd-meta"><span class="{chip_cls}">{chip}</span><span class="bd-score">{escape(plus_num(mean))}{escape(contribution_unit)}</span></div>'
        rows.append(
            f'<div class="bd-row">'
            f'<div><div class="bd-lab">{escape(label)}</div><div class="bd-val">{escape(value)}</div></div>'
            f'<div class="bd-track"><div class="{fill_cls}" style="width:{width}%"></div></div>'
            f'{score_html}'
            f'</div>'
        )
    if include_other:
        other=other_value
        other_width=max(18,min(96,int(round(28+64*(abs(other)/max_abs if max_abs else 0)))))
        if abs(other)<1e-12:
            other_chip="贡献接近 0"
            other_chip_class="bd-chip neutral"
            other_fill_class="bd-fill neutral"
        elif is_ohss_local:
            other_chip="增加风险" if other>0 else "降低风险"
            other_chip_class="bd-chip neg" if other>0 else "bd-chip"
            other_fill_class="bd-fill neg" if other>0 else "bd-fill"
        elif is_candidate_local:
            other_chip="增加预测" if other>0 else "降低预测"
            other_chip_class="bd-chip" if other>0 else "bd-chip neg"
            other_fill_class="bd-fill" if other>0 else "bd-fill neg"
        else:
            other_chip="局部正向" if other>0 else "局部负向"
            other_chip_class="bd-chip" if other>0 else "bd-chip neg"
            other_fill_class="bd-fill" if other>0 else "bd-fill neg"
        other_score=f"{escape(plus_num(other))}{escape(contribution_unit)}"
        other_detail=f"其余 {len(detail_items)} 个录入变量的贡献合计"
        detail_scores=[as_float(item.get("mean_shap"),0.0) for item in detail_items]
        detail_rows=[]
        for item,detail_score in zip(detail_items,detail_scores):
            feature=str(item.get("feature",item.get("label","--")))
            label=str(item.get("label",feature))
            value=str(item.get("value_label") or feature_value_text(feature,patient))
            if is_ohss_local:
                risk_increase=as_float(item.get("mean_shap"),0.0)>0
                detail_chip="增加风险" if risk_increase else "降低风险"
                detail_chip_class="bd-mini-chip down" if risk_increase else "bd-mini-chip up"
            elif is_candidate_local:
                detail_up=detail_score>=0
                detail_chip="增加预测" if detail_up else "降低预测"
                detail_chip_class="bd-mini-chip up" if detail_up else "bd-mini-chip down"
            else:
                detail_up=detail_score>=0
                detail_chip="局部正向" if detail_up else "局部负向"
                detail_chip_class="bd-mini-chip up" if detail_up else "bd-mini-chip down"
            detail_score_text=f"{detail_score:+.3f}{contribution_unit}"
            detail_rows.append(
                f'<div class="bd-other-item">'
                f'<div><div class="bd-other-name">{escape(label)}</div><div class="bd-other-value">{escape(value)}</div></div>'
                f'<span class="{detail_chip_class}">{detail_chip}</span>'
                f'<strong>{escape(detail_score_text)}</strong>'
                f'</div>'
            )
        detail_body="".join(detail_rows) if detail_rows else '<div class="bd-other-empty">暂无可展开的单项归因</div>'
        rows.append(
            f'<details class="bd-other-details">'
            f'<summary class="bd-row other">'
            f'<div><div class="bd-lab">其他因素总和</div><div class="bd-val">{escape(other_detail)}</div></div>'
            f'<div class="bd-track"><div class="{other_fill_class}" style="width:{other_width}%"></div></div>'
            f'<div class="bd-meta"><span class="{other_chip_class}">{other_chip}</span><span class="bd-score">{other_score}</span><span class="bd-other-toggle"><span class="closed">展开明细</span><span class="opened">收起明细</span></span></div>'
            f'</summary>'
            f'<div class="bd-other-list">{detail_body}</div>'
            f'</details>'
        )
    if is_ohss_local:
        note="严格中重度 OHSS 候选模型正类的校准后 Tree SHAP 贡献；青绿色和负号降低风险，红色和正号增加风险。仅展示患者页面真实录入字段，不使用隐藏项补齐预测概率。"
    elif is_candidate_local:
        note=f"展示贡献最高的 {MAIN_LOCAL_SHAP_LIMIT} 个当前促排录入字段；条形长度为当前 V2 candidate-response 模型的 Tree SHAP 贡献，青绿色增加预测，红色降低预测。仅汇总其余真实录入字段，不使用隐藏项补齐结果。"
    elif is_local:
        note=f"仅展示当前页面真实录入字段；数值为条件背景到当前推荐类别概率的积分梯度贡献（pp），青绿色为局部正向，红色为局部负向。非录入模型上下文在基准与当前患者间保持不变，不参与贡献补齐。"
    else:
        note=f"\u5c55\u793a\u8d21\u732e\u6700\u9ad8\u7684 {MAIN_LOCAL_SHAP_LIMIT} \u4e2a\u60a3\u8005\u5b57\u6bb5\uff1b\u6761\u5f62\u957f\u5ea6\u8868\u793a\u5f53\u524d\u60a3\u8005\u5b57\u6bb5\u5339\u914d\u5230\u7684\u76f8\u5bf9 SHAP \u8d21\u732e\u5f3a\u5ea6\uff0c\u9752\u7eff\u8272\u4e3a\u5e73\u5747\u6b63\u5411\u8d21\u732e\uff0c\u7ea2\u8272\u4e3a\u5e73\u5747\u8d1f\u5411\u8d21\u732e\u3002"
    badge=escape(str(badge_label)) if badge_label is not None else f"{float(probability)*100:.1f}%"
    if baseline_probability is not None:
        baseline_pct=clamp(as_float(baseline_probability,0.0),0.0,1.0)*100
        prediction_pct=clamp(as_float(probability,0.0),0.0,1.0)*100
        probability_display=(f'<div class="bd-prob-flow"><div class="bd-base" title="保持非录入模型上下文为患者当前值，仅将真实录入字段置于匹配训练背景后的条件预测"><span>基准预测值</span><strong>{baseline_pct:.1f}%</strong></div><span class="bd-arrow">&#8594;</span><span class="chip cp">{prediction_pct:.1f}%</span></div>')
    elif baseline_display is not None:
        probability_display=(f'<div class="bd-prob-flow"><div class="bd-base" title="保持非录入模型上下文和候选剂量为当前情景，仅将真实录入字段置于训练参考值后的条件预测"><span>基准预测值</span><strong>{escape(str(baseline_display))}</strong></div><span class="bd-arrow">&#8594;</span><span class="chip cp">{badge}</span></div>')
    else:
        probability_display=f'<span class="chip cp">{badge}</span>'
    card_class=" probability-card" if baseline_probability is not None else " outcome-card" if show_other else ""
    return f'<div class="bd-card{card_class}"><div class="bd-head"><div><div class="bd-title">{escape(str(drug))} \u4e2a\u4f53\u5316\u8d21\u732e</div><div class="bd-target">\u9884\u6d4b\u76ee\u6807: {escape(str(prediction_label))}</div></div>{probability_display}</div><div class="bd-stack">{"".join(rows)}</div></div>'


def technical_shap_details(local_targets):
    if not isinstance(local_targets, Mapping):
        return ""
    cards=[]
    for key in ("fsh","lh","hmg"):
        target=local_targets.get(key,{})
        items=[item for item in target.get("technical_items",[]) if isinstance(item,Mapping)]
        if not items:
            continue
        rows=[]
        for item in items[:4]:
            label=str(item.get("label",item.get("feature","--")))
            value=str(item.get("value_label","--"))
            mean=float(item.get("mean_shap",0.0))
            neg=mean<0
            rows.append(
                f'<div class="tech-row">'
                f'<div><div class="tech-label">{escape(label)}</div><div class="tech-value">{escape(value)}</div></div>'
                f'<span class="chip {"cd" if neg else "ct"}">{"局部负向" if neg else "局部正向"}</span>'
                f'<div class="tech-score {"neg" if neg else ""}">{escape(plus_num(mean))}</div>'
                f'</div>'
            )
        cards.append(
            f'<div class="tech-card"><div class="tech-title">{escape(str(target.get("drug",key.upper())))} · sample {escape(str(target.get("sample_id","--")))}</div>{"".join(rows)}</div>'
        )
    if not cards:
        return ""
    return (
        '<details class="tech-shap">'
        '<summary><span>模型派生特征（技术详情）</span><span class="chip cm">默认收起</span></summary>'
        f'<div class="tech-grid">{"".join(cards)}</div>'
        '<div class="bd-note" style="padding:0 16px 16px;margin-top:0">主界面优先展示医生录入字段；这里保留未在主卡展示的高贡献派生特征，便于技术复核。</div>'
        '</details>'
    )

def shap_page():
    title("\u4e0b\u4e00\u6b21 Gn \u5242\u91cf\u4e0e\u7ed3\u5c40\u98ce\u9669\u89e3\u91ca","解释当前患者快照下 FSH、LH、HMG 推荐类别、预测获卵数及严格中重度 OHSS 风险的真实模型归因。")
    patient_ctx=refresh_page_recommendations("\u63a8\u8350\u89e3\u91ca")
    if not recommendation_required_notice("\u63a8\u8350\u89e3\u91ca"):
        return
    b=best(auto_recompute=False); fl,fr=b["fsh_category"]; ll,lr=b["lh_category"]; hl,hr=b["hmg_category"]
    shap_data={"is_real":False,"targets":{},"groups":[]}
    if UI_REAL_DATA_AVAILABLE and load_phase867_dose_shap_summary is not None:
        try:
            shap_data=load_phase867_dose_shap_summary(limit=80)
        except Exception as exc:
            st.session_state["shap_ui_error"]=str(exc)
    targets=shap_data.get("targets",{}) if isinstance(shap_data,Mapping) else {}
    fsh_all=targets.get("fsh",{}).get("items") or [("E2 level","\u539f\u578b\u5360\u4f4d",78,"t","ct"),("\u226514 mm follicle count","\u539f\u578b\u5360\u4f4d",62,"t","ct"),("previous FSH dose","\u539f\u578b\u5360\u4f4d",56,"","cp"),("AMH","\u539f\u578b\u5360\u4f4d",38,"w","cw")]
    lh_all=targets.get("lh",{}).get("items") or [("serum LH","\u539f\u578b\u5360\u4f4d",70,"t","ct"),("E2 change","\u539f\u578b\u5360\u4f4d",48,"w","cw"),("monitoring visit order","\u539f\u578b\u5360\u4f4d",43,"","cp"),("P level","\u539f\u578b\u5360\u4f4d",38,"t","ct")]
    hmg_all=targets.get("hmg",{}).get("items") or [("previous HMG dose","\u539f\u578b\u5360\u4f4d",68,"","cp"),("mean follicle diameter","\u539f\u578b\u5360\u4f4d",56,"t","ct"),("\u226518 mm follicle count","\u539f\u578b\u5360\u4f4d",45,"t","ct"),("BMI","\u539f\u578b\u5360\u4f4d",34,"w","cw")]
    local_shap={"is_local":False,"targets":{}}
    if UI_REAL_DATA_AVAILABLE and load_phase867_local_dose_shap_for_patient is not None:
        try:
            local_shap=load_phase867_local_dose_shap_for_patient(patient_ctx, limit=MAIN_LOCAL_SHAP_LIMIT)
        except Exception as exc:
            st.session_state["local_shap_ui_error"]=str(exc)
    local_targets=local_shap.get("targets",{}) if isinstance(local_shap,Mapping) else {}
    fsh_breakdown=ui_input_contribution_items(local_targets.get("fsh",{}).get("items") or patient_breakdown_items("fsh", fsh_all, patient_ctx))
    lh_breakdown=ui_input_contribution_items(local_targets.get("lh",{}).get("items") or patient_breakdown_items("lh", lh_all, patient_ctx))
    hmg_breakdown=ui_input_contribution_items(local_targets.get("hmg",{}).get("items") or patient_breakdown_items("hmg", hmg_all, patient_ctx))
    sel=as_float(b.get("selection_rate"),.54); succ=as_float(b.get("success_rate"),.70); score=as_float(b.get("score"),.62); safe=1-as_float(b.get("safety_probability"),.08)
    dose_baseline_data={"baselines":{}}
    if UI_REAL_DATA_AVAILABLE and load_phase870_dose_probability_baselines is not None:
        try:
            dose_baseline_data=load_phase870_dose_probability_baselines()
        except Exception as exc:
            st.session_state["dose_probability_baseline_error"]=str(exc)
    dose_baselines=dose_baseline_data.get("baselines",{}) if isinstance(dose_baseline_data,Mapping) else {}
    dose_predictions=((b.get("dose_model_context") or {}).get("predictions") or {}) if isinstance(b,Mapping) else {}
    current_dose_attributions={}
    if build_current_patient_dose_attribution_items is not None:
        for drug in ("fsh","lh","hmg"):
            prediction=dose_predictions.get(drug,{}) if isinstance(dose_predictions,Mapping) else {}
            rows=build_current_patient_dose_attribution_items(
                prediction.get("local_attributions") or [],patient_ctx
            )
            if rows:
                current_dose_attributions[drug]=rows
    fsh_breakdown=current_dose_attributions.get("fsh") or fsh_breakdown
    lh_breakdown=current_dose_attributions.get("lh") or lh_breakdown
    hmg_breakdown=current_dose_attributions.get("hmg") or hmg_breakdown
    def dose_explanation_values(drug, fallback_display, fallback_probability):
        prediction=dose_predictions.get(drug,{}) if isinstance(dose_predictions,Mapping) else {}
        label=str(prediction.get("label", ""))
        display=str(prediction.get("display") or fallback_display)
        probability=_as_prob(prediction.get("probability"))
        probability=clamp(fallback_probability,0.0,1.0) if probability is None else probability
        baseline=_as_prob(prediction.get("attribution_baseline_probability"))
        if baseline is None:
            baseline=_as_prob((dose_baselines.get(drug) or {}).get(label))
        attribution=local_targets.get(drug,{}) if isinstance(local_targets,Mapping) else {}
        return display,probability,baseline,attribution.get("all_shap_abs_sum")
    fsh_display,fsh_probability,fsh_baseline,fsh_shap_abs_sum=dose_explanation_values("fsh",fl,clamp((sel+score)/2,.05,.95))
    lh_display,lh_probability,lh_baseline,lh_shap_abs_sum=dose_explanation_values("lh",ll,clamp((sel+succ)/2,.05,.95))
    hmg_display,hmg_probability,hmg_baseline,hmg_shap_abs_sum=dose_explanation_values("hmg",hl,clamp((safe+score)/2,.05,.95))
    ohss_prof=ohss_profile(b)
    ohss_target=f"中重度 OHSS 风险 / {ohss_prof['category_zh']}"
    oocyte_explanation={}; ohss_explanation={}
    snap=None
    if CANDIDATE_RESPONSE_AVAILABLE and explain_candidate_response_shap is not None and patient_form_to_snapshot is not None:
        try:
            snap=patient_form_to_snapshot(patient_ctx)
            efficacy_run=current_efficacy_run_id() if current_efficacy_run_id is not None else None
            safety_run=current_run_id() if current_run_id is not None else None
            common=dict(
                fsh_dose=b["fsh"],
                lh_dose=b["lh"],
                hmg_dose=b["hmg"],
                limit=256,
                attribution_features=set(UI_INPUT_ATTRIBUTION_FEATURES),
            )
            oocyte_explanation=explain_candidate_response_shap(snap,task="layer2_oocytes",run_id=efficacy_run,**common)
            ohss_explanation=explain_candidate_response_shap(snap,task="layer2_ohss",run_id=safety_run,**common)
        except Exception as exc:
            st.session_state["candidate_response_explain_error"]=str(exc)
    oocyte_all=candidate_shap_breakdown_items(oocyte_explanation,patient_ctx)
    ohss_all=candidate_shap_breakdown_items(ohss_explanation,patient_ctx,probability=True)
    oocyte_breakdown=[item for item in oocyte_all if item.get("clinical_display")]
    ohss_breakdown=[item for item in ohss_all if item.get("clinical_display")]
    gn_cards=[
        breakdown_card("FSH",f"FSH = {fsh_display}",fsh_probability,fsh_breakdown,patient_ctx,baseline_probability=fsh_baseline,other_items=fsh_breakdown[MAIN_LOCAL_SHAP_LIMIT:],show_other=True,contribution_unit=" pp"),
        breakdown_card("LH",f"LH = {lh_display}",lh_probability,lh_breakdown,patient_ctx,baseline_probability=lh_baseline,other_items=lh_breakdown[MAIN_LOCAL_SHAP_LIMIT:],show_other=True,contribution_unit=" pp"),
        breakdown_card("HMG",f"HMG = {hmg_display}",hmg_probability,hmg_breakdown,patient_ctx,baseline_probability=hmg_baseline,other_items=hmg_breakdown[MAIN_LOCAL_SHAP_LIMIT:],show_other=True,contribution_unit=" pp"),
    ]
    outcome_cards=[]
    if oocyte_breakdown:
        oocyte_baseline=oocyte_explanation.get("baseline_prediction")
        oocyte_prediction=oocyte_explanation.get("prediction",b.get("o"))
        outcome_cards.append(breakdown_card("\u83b7\u5375\u6570",f"\u9884\u6d4b\u83b7\u5375\u6570 = {fmt(oocyte_prediction)}",0,oocyte_breakdown,patient_ctx,badge_label=f"{fmt(oocyte_prediction)} \u4e2a",baseline_display=f"{oocyte_baseline:.1f} \u4e2a" if oocyte_baseline is not None else "--",other_items=oocyte_breakdown[MAIN_LOCAL_SHAP_LIMIT:],show_other=True,contribution_unit=" \u4e2a"))
    if ohss_breakdown:
        ohss_baseline=ohss_explanation.get("baseline_prediction")
        ohss_prediction=clamp(as_float(ohss_explanation.get("prediction"),ohss_display_profile(b).get("display") or 0),0,1)
        outcome_cards.append(breakdown_card("OHSS",ohss_target,ohss_prediction,ohss_breakdown,patient_ctx,badge_label=f"{ohss_prediction*100:.2f}%",baseline_display=f"{ohss_baseline*100:.2f}%" if ohss_baseline is not None else "--",other_items=ohss_breakdown[MAIN_LOCAL_SHAP_LIMIT:],show_other=True,contribution_unit=" pp"))
    gn_breakdown_html="".join(gn_cards)
    outcome_breakdown_html="".join(outcome_cards)
    outcome_html=(f'<div class="breakdown-row-title outcome-title">\u7ed3\u5c40\u4e0e\u98ce\u9669\u4e2a\u4f53\u5316\u89e3\u91ca</div><div class="breakdown-grid outcome-explain-grid" style="grid-template-columns:repeat(2,minmax(0,1fr))">{outcome_breakdown_html}</div>') if outcome_cards else ""
    technical_html=""
    if local_shap.get("is_local"):
        matched_meta=" | ".join(
            f'{local_targets.get(key,{}).get("drug",key.upper())} sample {local_targets.get(key,{}).get("sample_id","--")}'
            for key in ("fsh","lh","hmg")
            if local_targets.get(key,{})
        )
        local_chip='<span class="chip ct">\u771f\u5b9e\u5c40\u90e8\u5f52\u56e0</span>'
        local_density_chip=f'<span class="chip cm">Top {MAIN_LOCAL_SHAP_LIMIT} \u4e34\u5e8a\u5b57\u6bb5</span>'
        local_notice=f'\u5df2\u63a5\u5165 Phase 8.67 per-sample attribution long \u8868\uff1b\u5f53\u524d\u60a3\u8005\u5feb\u7167\u4f1a\u5339\u914d\u6700\u63a5\u8fd1\u7684\u771f\u5b9e OOF \u5c40\u90e8\u5f52\u56e0\u6837\u672c\uff0c\u5e76\u7528\u8be5\u6837\u672c\u5b9e\u9645 shap_value \u6e32\u67d3 Gn \u5242\u91cf\u8fdb\u5ea6\u6761\u3002{escape(matched_meta)}'
    else:
        local_chip='<span class="chip cw">\u5168\u5c40\u6c47\u603b\u56de\u9000</span>'
        local_density_chip=""
        local_notice='\u672a\u8bfb\u53d6\u5230\u53ef\u5339\u914d\u7684 per-sample attribution long \u8868\uff0c\u5f53\u524d Gn \u5242\u91cf\u89e3\u91ca\u56de\u9000\u4e3a\u60a3\u8005\u5b57\u6bb5 + \u5168\u5c40 SHAP \u6c47\u603b\u5f3a\u5ea6\u3002'
    ohss_chip='<span class="chip cw">\u4e2d\u91cd\u5ea6 OHSS \u65e9\u671f\u9884\u8b66</span>' if ohss_breakdown else ''
    st.markdown(f'<div class="sec"><div class="head"><div class="ic">&plusmn;</div><h3>\u4e2a\u4f53\u5316\u8d21\u732e</h3></div><div class="breakdown-row-title">Gn \u5242\u91cf\u4e2a\u4f53\u5316\u89e3\u91ca</div><div class="breakdown-grid gn-explain-grid">{gn_breakdown_html}</div>{outcome_html}{technical_html}</div>',unsafe_allow_html=True)
    knn_evidence_panel()

def main():
    init(); header(); p=st.session_state.page
    if p=="首页": home()
    elif p=="患者录入": patient_page()
    elif p=="决策曲线": knn_page()
    elif p=="推荐解释": shap_page()
    elif p=="监测结果": rec_page(True)
    statusbar(); st.markdown('<div style="height:42px"></div>',unsafe_allow_html=True)

if __name__ == "__main__": main()
