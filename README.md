# Predicting Ovarian UI

Standalone Streamlit prototype for the IVF/ICSI GnRH-a ultra-long protocol Gn dose decision-support UI.

## 师弟使用步骤

推荐使用 Python 3.10 或 3.11。不要直接用未固定依赖的 Python 3.12 base 环境运行，因为仓库里的部分模型文件由 scikit-learn 1.3.x 生成，Python 3.12 环境可能出现模型 pickle 兼容问题。

### 1. 下载代码

推荐用 Git 克隆，后续方便修改和提交：

```bash
git clone https://github.com/Haaan1011/Predicting_Ovarian_UI.git
cd Predicting_Ovarian_UI
```

如果只是查看界面，也可以在 GitHub 页面点击 `Code` -> `Download ZIP`，解压后进入项目文件夹。

### 2. 创建环境并安装依赖

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Linux/macOS：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果电脑上已经有原项目环境，也可以直接使用：

```bash
conda activate Han_Overian
```

### 3. 启动界面

```bash
streamlit run prototype/streamlit_app/app.py --server.address 127.0.0.1 --server.port 18501
```

启动后在浏览器打开：

```text
http://127.0.0.1:18501/?view=knn
```

常用页面地址：

- 患者录入页：`http://127.0.0.1:18501/?view=input`
- 监测结果页：`http://127.0.0.1:18501/?view=monitor`
- 决策曲线页：`http://127.0.0.1:18501/?view=knn`
- 推荐解释页：`http://127.0.0.1:18501/?view=shap`

### 4. 修改 UI 代码

主要修改下面几个文件：

```text
prototype/streamlit_app/app.py
prototype/streamlit_app/ui_real_data_sources.py
prototype/streamlit_app/candidate_response_service.py
prototype/streamlit_app/dose_recommendation_service.py
```

一般只改页面布局、文案、样式时，优先改 `prototype/streamlit_app/app.py`。不要改论文主项目、训练数据、论文结果表或模型训练脚本。

### 5. 修改后自检

```bash
python -m py_compile prototype/streamlit_app/app.py prototype/streamlit_app/ui_real_data_sources.py prototype/streamlit_app/dose_recommendation_service.py prototype/streamlit_app/candidate_response_service.py
python scripts/qa/smoke_check_ui_models.py
```

自检通过后再启动 Streamlit 看页面。

### 6. 提交修改

```bash
git status
git add prototype/streamlit_app/app.py prototype/streamlit_app/ui_real_data_sources.py
git commit -m "update UI"
git push
```

如果只下载 ZIP 而不是 `git clone`，可以本地修改和运行，但不能直接 `git push` 回 GitHub。

## What is included

- Streamlit clinician UI: patient input, monitoring result, decision curves, recommendation explanation.
- Current model integration needed by the UI: UI-reduced GRU(AddGate) dose models, candidate oocyte/OHSS response bundles, local SHAP support tables, and KNN support artifacts.
- De-identified runtime tables required for KNN/background statistics.

## What is intentionally not included

- Raw source spreadsheets or direct patient identifiers.
- Manuscript drafts, paper figure workflows, and unrelated training outputs.
- Downstream outcome UI outputs that are outside the current prototype scope. The prototype currently displays predicted oocytes and strict moderate-to-severe OHSS risk only.

## Run

Recommended runtime: Python 3.10 or 3.11. The bundled model artifacts were produced with scikit-learn 1.3.x, so avoid running them in an unpinned Python 3.12 base environment.

Use the project conda environment if available:

```bash
conda activate Han_Overian
streamlit run prototype/streamlit_app/app.py --server.address 127.0.0.1 --server.port 18501
```

Or create a fresh environment and install dependencies.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run prototype/streamlit_app/app.py --server.address 127.0.0.1 --server.port 18501
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run prototype/streamlit_app/app.py --server.address 127.0.0.1 --server.port 18501
```

Open the main decision-curve page:

http://127.0.0.1:18501/?view=knn

Other useful pages:

- Patient input: http://127.0.0.1:18501/?view=input
- Monitoring result: http://127.0.0.1:18501/?view=monitor
- Recommendation explanation: http://127.0.0.1:18501/?view=shap

## Smoke check

```bash
python scripts/qa/smoke_check_ui_models.py
```

## Clinical boundary

This is a clinical decision-support prototype, not an automatic order-entry system. The dose output should be described as model-recommended dose or candidate-dose scenario analysis, and final medication decisions remain with the clinician. SHAP is model attribution; KNN is similar-case evidence; dose-response curves are conditional scenario predictions, not causal effects.
