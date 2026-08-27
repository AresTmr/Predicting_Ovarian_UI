#!/usr/bin/env bash
set -euo pipefail
PORT=
streamlit run prototype/streamlit_app/app.py --server.address 127.0.0.1 --server.port  
