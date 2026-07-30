# Paper feature list — Cohen, Lu & Nguyen (2025), *Mimicking Finance*

Source: `Mimicking-Finance---Nov-2025.pdf`, §3.2 (Features and Target Variable
Construction) and Appendix A (Variable Descriptions). All features are **lagged** so they
are observable at prediction time.

## Prediction target (not a feature)

3-class next-quarter trade direction at the fund–security–quarter level:

$$Y_{i,t}=\begin{cases}-1 & \Delta sh_{i,t}\le -0.01 \quad(\text{trim/sell})\\ \;\;0 & |\Delta sh_{i,t}|<0.01 \quad(\text{no material change})\\ +1 & \Delta sh_{i,t}\ge 0.01 \quad(\text{add/buy})\end{cases},\qquad \Delta sh_{i,t}=\frac{sh_{i,t+1}-sh_{i,t}}{sh_{i,t}+1}$$

A $\pm1\%$ band around zero counts as "no change."

## Feature groups (LSTM inputs)

### 1. Fund-level signals
- **Lagged Fund Returns** — prior-quarter and multi-quarter returns of the fund portfolio.
- **Lagged Fund Flows** — net cash inflow/outflow at the fund level, aggregated to the quarter.
- **Lagged Fund Characteristics** — fund size (AUM), value tilt, and momentum tilt, constructed from holdings or reported style metrics.

### 2. Security-level signals
- **Lagged Security Characteristics** — equity size ($\log$ market cap), value (book-to-market), and momentum (past returns over standard lookbacks).
- **Risk/Factor Exposures** — curated factor list $\{\text{market},\ \text{size},\ \text{value},\ \text{profitability},\ \text{investment},\ \text{short-term reversal}\}$, merged at the permno–month level and aligned to quarters.

### 3. Peer / category behavior
- **Category Activity Rates** — share-increase, share-decrease, and no-change frequencies of peer funds within the fund's Morningstar style category, plus their lags.

### 4. Macroeconomic backdrop
- **Lagged Macroeconomic Factors** — level and slope of the yield curve and credit conditions: term spreads (e.g. $5y\!-\!1y$, $10y\!-\!3m$), default spreads (e.g. BAA–Treasury), and related short/long yields, both real and nominal.

### 5. Position & within-fund context
- **Market Value and Shares** — current position market value $mv$, and lagged share counts $sh$ tracked up to **six quarters**.
- **Weight Dynamics** — lagged and pass-through values used to compute portfolio-weight changes.
- **Rank Inside Fund** ($id$) — size rank of the security within the fund each quarter ($1=$ largest holding).
- **Padding Indicator** ($mask$) — flags placeholder rows created to keep the balanced panel rectangular.

## Granular characteristics (Appendix A glossary)

These named characteristics populate the "Lagged Fund/Security Characteristics" buckets above:

| Variable | Description |
|---|---|
| Actual 12b-1 | Annual distribution (12b-1) fee charged by the fund. |
| Asset Turnover | Ratio of sales to total assets. |
| Book Equity/ME | Ratio of book equity to market equity. |
| CAPM Idiosyncratic Vol. | Std. dev. of residuals from a CAPM regression over the prior 36 months. |
| Earnings Surprise | Reported quarterly earnings minus most recent consensus forecast, scaled by price. |
| Expense Ratio | Ratio of total annual expenses to average net assets. |
| Firm Age | Months since the firm's first appearance in CRSP. |
| Free Cash Flow/ME | Free cash flow scaled by market equity. |
| Fund Age | Quarters since the fund's inception. |
| Fund Flow | Net quarterly flow into the fund, scaled by total net assets. |
| Gross Profit/ME | Gross profits scaled by market equity. |
| Has Sales Restrictions | =1 if the fund imposes sales restrictions (e.g. redemption fees). |
| Income Yield | Annualized dividend yield of the fund. |
| Management Fee | Annual management fee charged by the fund. |
| Manager #Funds | Avg. number of funds simultaneously managed by a given manager. |
| Manager #Styles | Avg. number of distinct investment styles overseen by a manager. |
| Manager Ownership | Avg. dollar ownership of fund managers in the fund. |
| Manager Tenure (Max) | Maximum manager tenure (quarters) among all managers of a fund. |
| Manager Tenure (Mean) | Average manager tenure (quarters) in a fund. |
| Market Equity | Market value of the firm's equity ($ millions). |
| Momentum 1–6 Months | Cumulative stock return, months 1–6 prior to formation. |
| Momentum 1–12 Months | Cumulative stock return, months 1–12 prior to formation. |
| MVRank | Rank of a security's market value within a fund's portfolio (higher = larger). |
| NAV/52-week-high NAV | Current NAV relative to highest NAV in the prior 52 weeks. |
| Net Equity Payout/ME | Repurchases − issuances + dividends, scaled by market equity. |
| Number Funds | Number of distinct funds holding the security in a given quarter. |
| Number of Managers | Count of individual portfolio managers of the fund. |
| Open to Investment | =1 if the fund is open to new investors. |
| R&D/Sales | R&D expenditures scaled by sales. |
| Sales/BE | Sales scaled by book equity. |
| Short Term Reversal | Past one-month stock return (negative = recent losers). |
| Total Debt/ME | Ratio of total debt to market equity. |
| Total Net Asset | Total market value of a fund's assets ($ millions). |
| Turnover Ratio | Proportion of a fund's portfolio replaced during the year. |
| Within Category Competition | Number of other funds in the same Morningstar category that quarter. |
| Within Management Company Competition | Number of other funds managed by the same management company that quarter. |

**Caveat:** the same Appendix A glossary also lists $CRET_{0,1\ldots4}$ (cumulative excess
fund returns 1–4 quarters ahead), *Prediction Precision*, and *Naive Prediction Precision* —
these are **outcomes/targets, not inputs**. Do not treat the whole glossary as the feature vector.

## Mapping to this repository

The paper assumes CRSP/Compustat factor & macro panels. The single-parquet pipeline here
implements groups **1, 3, 5** plus whatever security fields the file carries, and omits the
external **factor (group 2 Risk/Factor Exposures)** and **macro (group 4)** blocks unless
those columns are present. See `README.md` / `DOCUMENTATION.md` for the exact `Config.features`
used (`weight, w_lag1, dw, rank, log_posval, log_pv, log_mktcap, quarterly_ret, past_1q_ret,
pdsh, pdsh_sign, pdsh_lag1, sh_lag1..3, peer_buy/sell/hold, n_holdings, fund_ret_l1`).
