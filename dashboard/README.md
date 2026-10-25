# TERRAIN Streamlit dashboard

Install and run from the repository root:

```powershell
python -m pip install -r dashboard/requirements.txt
python -m streamlit run dashboard/app.py
```

The dashboard checks `data/processed` for each configured city's GeoPackage.
Available files are visualized directly. Missing cities show an unavailable state and the expected input path; the dashboard never substitutes representative values.
