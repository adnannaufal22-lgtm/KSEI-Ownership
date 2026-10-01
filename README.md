# KSEI Ownership Dashboard

A focused Streamlit dashboard for monthly Indonesian stock-ownership data and daily holdings above 5% published by BEI/KSEI.

The interface uses a responsive, information-dense capital-markets terminal layout with native Streamlit controls and interactive Plotly charts. The presentation layer is separate from the BEI data-loading and ownership-calculation logic.

Live dashboard: https://ksei-ownership-dashboard.streamlit.app/

## Open the dashboard on this computer

Double-click **Open Dashboard.bat**. The first launch creates the local environment and installs the required packages. Later launches open the dashboard directly; terminal commands are not required.

## Dashboard structure

Choose one Stock or one Owner from the selectors in the dashboard header. That selection drives the entity-specific tabs, while the two market-wide movement tabs scan the complete ownership datasets:

1. **1% Ownership** — holder/stock history chart, number-of-shares pivot, and ownership-percentage pivot.
2. **5% Ownership** — daily beneficial-owner monitoring with separate Beneficial Owner, Latest Account Position, and Account Movement views.
3. **Classification** — dynamic investor-classification history and matching pivots.
4. **Type** — scrip versus scripless and Domestic versus Foreign charts with their pivots placed alongside.
5. **Entity Movement** — period-over-period movement charts for the selected stock or owner.
6. **Monthly Changes** — month selector, compact market summary, Top 10/all stock and owner rankings, and clickable row drill-downs showing the counterparties behind each reported change.
7. **Daily >5% Movement** — market-wide daily scanner with ticker, beneficial owner, signal, securities-account holder, residency, and minimum-change filters.

Classification and Type are stock-level source datasets, so those tabs explain that individual owner analysis is unavailable when Analyze By is set to Owner.

## Data update

Add each new BEI/IDX workbook to its matching folder:

- `BEI_Data\1% Ownership`
- `BEI_Data\Classification`
- `BEI_Data\Type`
- `BEI_Data\5% Ownership`

Keep one workbook per month in each monthly folder. A filename containing `YYYY-MM` is recommended, for example `2026-09_1% Ownership.xlsx`. Put each downloaded IDX **Pemegang Saham di Atas 5%** daily workbook in `BEI_Data\5% Ownership`; overlapping observations are deduplicated automatically and the newest source is retained when IDX republishes a date.

After adding the files, double-click **Publish Monthly Update.bat** to validate, commit, and push the changed data. Streamlit Community Cloud rebuilds the hosted app from the GitHub repository.

The dashboard automatically:

- detects BEI header rows even when their position changes;
- discovers future `.xlsx` or `.xlsm` files in all four folders;
- skips malformed or empty files in the running app while logging the issue;
- consolidates high-confidence monthly holder aliases using legal-name normalization and exact adjacent-month share continuity;
- keeps missing observations blank instead of turning them into zero;
- formats displayed share values with comma separators;
- shades monthly pivot cells light green, yellow, or red for increases, no change, or decreases;
- adds a month-over-month percentage-change heatmap below every monthly number-of-shares pivot;
- provides clickable holder links that switch the global analysis to that owner;
- dynamically detects the previous/current dates in each daily >5% workbook;
- separates `Nama Pemegang Saham` (beneficial owner) from `Nama Pemegang Rekening Efek` and `Nama Rekening Efek` (account position);
- classifies accumulation, selling, unchanged ownership, threshold entries/exits, and internal transfers between securities accounts;
- reconciles each beneficial-owner total to its underlying securities-account rows and exposes an audit log;
- caches parsed daily workbooks locally in `BEI_Data\.cache` so only new or changed source files need to be reparsed.

## Manual start and test

    python -m pip install -r requirements.txt
    python -m streamlit run app.py

For the parser test suite:

    python -m pip install -r requirements-dev.txt
    python -m pytest -q

The local address is normally http://localhost:8501.

## Source schema assumptions

- **1% Ownership:** reporting date, stock code, issuer name, owner name, total holding shares, and ownership percentage. Optional type, residency, nationality, domicile, scripless, and scrip columns are mapped when present.
- **Classification:** one stock row per month, identity columns plus any number of classification share columns and `TOTAL SCRIPLESS`. Classification columns are detected dynamically.
- **Type:** a three-row header containing `NUMBER OF SHARES`, Domestic/Foreign groups, investor-category and holding-band columns, and `TOTAL SCRIPLESS`.
- **Daily >5%:** two-row daily header containing stock code/name, `Nama Pemegang Saham`, combined holdings/percentage, and the two securities-account identity fields. Blank or omitted owners are treated as not reported, not automatically as a sale.

Column aliases for the monthly files are maintained in `schema_mapping.json`. In Type data, `Scrip Shares = Number of Shares - Total Scripless`; Domestic plus Foreign must reconcile to Total Scripless.
