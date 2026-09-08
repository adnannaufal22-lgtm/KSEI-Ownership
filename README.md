# KSEI Ownership Dashboard

Interactive Streamlit dashboard for monthly Indonesian stock ownership data published by BEI/KSEI.

## Open the dashboard on this computer

Double-click **Open Dashboard.bat**. The first launch installs the required packages once. Later launches open the dashboard directly.

## Monthly update

1. Download the new monthly 1% Ownership workbook from BEI.
2. Put it in the BEI_Data folder. Keep one workbook per month and use a filename containing YYYY-MM, for example 2026-09_1% Ownership.xlsx.
3. Double-click **Publish Monthly Update.bat**.

The update tool validates every monthly file, commits the changed BEI_Data folder to GitHub, and triggers the hosted Streamlit app to refresh.

The dashboard automatically:

- detects BEI disclaimer and header rows even when their position changes;
- combines every monthly .xlsx or .xlsm file in BEI_Data;
- rejects duplicate monthly files and filename/data-month mismatches;
- supports both INVESTOR_TYPE and INVESTOR_CLASSIFICATION;
- consolidates high-confidence holder aliases using legal-name normalization and exact adjacent-month share continuity;
- displays monthly holder/stock pivots with comma-separated share values;
- provides clickable holder links into the holder movement view.

## Manual start

    python -m pip install -r requirements.txt
    python -m streamlit run app.py

The local address is normally http://localhost:8501.

## Data source

The app reads BEI_Data directly. It no longer depends on a separately prepared Summary Ownership.xlsx.

Expected core fields are configured in schema_mapping.json: reporting date, stock code, issuer name, holder name, investor classification, local/foreign status, total shares, and ownership percentage.

