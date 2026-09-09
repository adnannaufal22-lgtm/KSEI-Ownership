# KSEI Ownership Dashboard

A focused Streamlit dashboard for monthly Indonesian stock-ownership data published by BEI/KSEI.

Live dashboard: https://ksei-ownership-dashboard.streamlit.app/

## Open the dashboard on this computer

Double-click **Open Dashboard.bat**. The first launch creates the local environment and installs the required packages. Later launches open the dashboard directly; terminal commands are not required.

## Dashboard structure

Choose one global entity in the sidebar: either one Stock or one Owner. That selection drives exactly four tabs:

1. **1% Ownership** — holder/stock history chart, number-of-shares pivot, and ownership-percentage pivot.
2. **Classification** — dynamic investor-classification history and matching pivots.
3. **Type** — scrip versus scripless and Domestic versus Foreign charts with their pivots placed alongside.
4. **Monthly Change** — period-over-period movement charts without a duplicate table.

Classification and Type are stock-level source datasets, so those tabs explain that individual owner analysis is unavailable when Analyze By is set to Owner.

## Monthly update

Add each new BEI workbook to its matching folder:

- `BEI_Data\1% Ownership`
- `BEI_Data\Classification`
- `BEI_Data\Type`

Keep one workbook per month in each folder. A filename containing `YYYY-MM` is recommended, for example `2026-09_1% Ownership.xlsx`. The app uses the actual reporting date inside the workbook and uses the filename month only as a fallback where supported.

After adding the files, double-click **Publish Monthly Update.bat**. It validates all three datasets, commits the changed `BEI_Data` folder to GitHub, and triggers the hosted Streamlit app to refresh.

The dashboard automatically:

- detects BEI header rows even when their position changes;
- discovers future monthly `.xlsx` or `.xlsm` files in all three folders;
- skips malformed or empty files in the running app while logging the issue;
- blocks publication if a monthly source file cannot be validated;
- consolidates high-confidence holder aliases using legal-name normalization and exact adjacent-month share continuity;
- keeps missing monthly observations blank instead of turning them into zero;
- formats displayed share values with comma separators;
- shades pivot cells light green, yellow, or red for increases, no change, or decreases;
- provides clickable holder links that switch the global analysis to that owner.

## Manual start

    python -m pip install -r requirements.txt
    python -m streamlit run app.py

The local address is normally http://localhost:8501.

## Source schema assumptions

- **1% Ownership:** reporting date, stock code, issuer name, owner name, total holding shares, and ownership percentage. Optional type, residency, nationality, domicile, scripless, and scrip columns are mapped when present.
- **Classification:** one stock row per month, identity columns plus any number of classification share columns and `TOTAL SCRIPLESS`. Classification columns are detected dynamically.
- **Type:** a three-row header containing `NUMBER OF SHARES`, Domestic/Foreign groups, investor-category and holding-band columns, and `TOTAL SCRIPLESS`.

Column aliases are maintained in `schema_mapping.json`. In Type data, `Scrip Shares = Number of Shares - Total Scripless`; Domestic plus Foreign must reconcile to Total Scripless.
