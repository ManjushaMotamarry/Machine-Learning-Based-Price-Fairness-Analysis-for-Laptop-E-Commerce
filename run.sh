#!/bin/bash
python -m venv venv
source venv/bin/activate
pip install -r streamlit_app/requirements.txt
cd streamlit_app
python -m streamlit run app.py