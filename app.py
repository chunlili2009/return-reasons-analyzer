import streamlit as st
import pandas as pd
import os
from openai import OpenAI
import json
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

st.set_page_config(page_title="Return Reason Analyzer", layout="wide")

st.title("Return Reason Analyzer")
st.write(
    "Upload a returns CSV with a `return_reason` column. "
    "Optionally upload an `orders.csv` file to calculate return rate by product."
)

# -----------------------------
# Onboarding / instructions
# -----------------------------
with st.expander("How to use this app", expanded=True):
    st.markdown(
        """
**Steps**
1. Upload a **Returns CSV** with at least a `return_reason` column.
2. Include `product` if you want product analysis and SKU diagnosis.
3. Include `date` if you want date filtering and trend analysis.
4. Optionally upload an **Orders CSV** with `product` and `orders` to calculate return rate.
5. Click **Analyze Returns**.
6. Start with the **SKU Diagnosis Table** and focus on high-priority SKUs first.

**Returns CSV recommended columns**
- `return_reason` (required)
- `product` (recommended)
- `date` (optional)

**Orders CSV required columns**
- `product`
- `orders`
"""
    )

# -----------------------------
# Sample downloads
# -----------------------------
sample_returns_df = pd.DataFrame(
    {
        "date": [
            "2026-03-01",
            "2026-03-01",
            "2026-03-02",
            "2026-03-03",
            "2026-03-03",
            "2026-03-04",
            "2026-03-04",
            "2026-03-05",
        ],
        "product": [
            "Dress A",
            "Dress A",
            "Shirt B",
            "Shirt B",
            "Pants C",
            "Dress A",
            "Dress A",
            "Shirt B",
        ],
        "return_reason": [
            "Too small",
            "Size runs smaller than chart",
            "Poor stitching quality",
            "Color difference from photos",
            "Arrived damaged",
            "Too tight in shoulders",
            "Not true to size",
            "Loose seam after one wear",
        ],
    }
)

sample_orders_df = pd.DataFrame(
    {
        "product": ["Dress A", "Shirt B", "Pants C"],
        "orders": [200, 150, 80],
    }
)

st.download_button(
    label="Download Sample Returns CSV",
    data=sample_returns_df.to_csv(index=False).encode("utf-8"),
    file_name="sample_returns.csv",
    mime="text/csv",
)

st.download_button(
    label="Download Sample Orders CSV",
    data=sample_orders_df.to_csv(index=False).encode("utf-8"),
    file_name="sample_orders.csv",
    mime="text/csv",
)

# -----------------------------
# Controls
# -----------------------------
analysis_mode = st.radio(
    "Choose analysis mode:",
    ["Overall Analysis", "Product Analysis", "Both"],
    horizontal=True,
)

st.markdown("### SKU Diagnosis Settings")
col_a, col_b = st.columns(2)
with col_a:
    min_returns_per_sku = st.number_input(
        "Minimum return records per SKU for diagnosis",
        min_value=1,
        max_value=20,
        value=2,
        step=1,
    )
with col_b:
    max_skus_to_diagnose = st.number_input(
        "Maximum number of SKUs to diagnose",
        min_value=1,
        max_value=100,
        value=10,
        step=1,
    )

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
    st.stop()

client = OpenAI(api_key=api_key)

returns_file = st.file_uploader("Upload Returns CSV", type="csv")
orders_file = st.file_uploader("Upload Orders CSV (optional)", type="csv")

# -----------------------------
# Helpers
# -----------------------------
def clean_json_response(result_text: str):
    if not result_text or not result_text.strip():
        return None, "The model returned an empty response."

    cleaned = result_text.strip()

    if cleaned.startswith("```json"):
        cleaned = cleaned.replace("```json", "", 1).strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```", "", 1).strip()

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()

    try:
        return json.loads(cleaned), None
    except Exception:
        return None, result_text


def run_data_quality_checks(df):
    original_row_count = len(df)

    missing_return_reason = 0
    missing_product = 0
    invalid_dates = 0
    duplicate_rows = 0

    working_df = df.copy()

    if "return_reason" in working_df.columns:
        rr_series = working_df["return_reason"].fillna("").astype(str).str.strip()
        missing_return_reason = int((rr_series == "").sum())
        working_df["return_reason"] = rr_series

    if "product" in working_df.columns:
        product_series = working_df["product"].fillna("").astype(str).str.strip()
        missing_product = int((product_series == "").sum())
        working_df["product"] = product_series

    if "date" in working_df.columns:
        original_date_non_null = working_df["date"].notna().sum()
        working_df["date"] = pd.to_datetime(working_df["date"], errors="coerce")
        valid_date_non_null = working_df["date"].notna().sum()
        invalid_dates = int(original_date_non_null - valid_date_non_null)

    if "return_reason" in working_df.columns:
        working_df = working_df[working_df["return_reason"] != ""]

    if "product" in working_df.columns:
        working_df.loc[working_df["product"] == "", "product"] = "Unknown Product"

    duplicate_rows = int(working_df.duplicated().sum())
    working_df = working_df.drop_duplicates()

    cleaned_row_count = len(working_df)
    rows_removed = original_row_count - cleaned_row_count

    summary = {
        "original_row_count": original_row_count,
        "missing_return_reason": missing_return_reason,
        "missing_product": missing_product,
        "invalid_dates": invalid_dates,
        "duplicate_rows_removed": duplicate_rows,
        "cleaned_row_count": cleaned_row_count,
        "rows_removed_total": rows_removed,
    }

    return working_df, summary


def render_data_quality_summary(summary):
    st.subheader("Data Quality Summary")

    col1, col2, col3 = st.columns(3)
    col1.metric("Original Rows", summary["original_row_count"])
    col2.metric("Rows Removed", summary["rows_removed_total"])
    col3.metric("Clean Rows Ready", summary["cleaned_row_count"])

    st.markdown("### Quality Checks")
    st.write(f"- Missing `return_reason`: **{summary['missing_return_reason']}**")
    st.write(f"- Missing `product`: **{summary['missing_product']}**")
    st.write(f"- Invalid `date` values: **{summary['invalid_dates']}**")
    st.write(f"- Duplicate rows removed: **{summary['duplicate_rows_removed']}**")

    issues_found = (
        summary["missing_return_reason"]
        + summary["missing_product"]
        + summary["invalid_dates"]
        + summary["duplicate_rows_removed"]
    )

    if issues_found == 0:
        st.success("No data quality issues were detected.")
    else:
        st.warning("Some data quality issues were detected and cleaned where possible.")


def load_and_clean_orders(orders_file):
    try:
        orders_df = pd.read_csv(orders_file)
    except Exception as e:
        st.error(f"Could not read orders CSV file: {str(e)}")
        return None

    required_columns = {"product", "orders"}
    if not required_columns.issubset(set(orders_df.columns)):
        st.error("Orders CSV must contain `product` and `orders` columns.")
        return None

    orders_df = orders_df.copy()
    orders_df["product"] = orders_df["product"].fillna("").astype(str).str.strip()
    orders_df["orders"] = pd.to_numeric(orders_df["orders"], errors="coerce")
    orders_df = orders_df[(orders_df["product"] != "") & (orders_df["orders"].notna())]
    orders_df = orders_df.drop_duplicates(subset=["product"], keep="last")

    return orders_df


def render_trend_section(df):
    if "date" not in df.columns:
        return

    st.subheader("Return Trend Over Time")

    trend_df = df.copy().dropna(subset=["date"])
    if trend_df.empty:
        st.write("No valid dates available for trend analysis.")
        return

    daily_counts = (
        trend_df.groupby(trend_df["date"].dt.date)
        .size()
        .reset_index(name="Return Count")
    )
    daily_counts.columns = ["Date", "Return Count"]

    days = len(daily_counts)
    total_returns = int(daily_counts["Return Count"].sum())
    avg_returns = float(daily_counts["Return Count"].mean())

    col1, col2, col3 = st.columns(3)
    col1.metric("Days in Filter", days)
    col2.metric("Total Returns", total_returns)
    col3.metric("Avg Returns / Day", round(avg_returns, 2))

    st.markdown("### Daily Return Volume")
    st.line_chart(daily_counts.set_index("Date"))

    st.markdown("### Daily Return Table")
    st.dataframe(daily_counts, use_container_width=True)


def render_word_cloud(df):
    if "return_reason" not in df.columns:
        return

    text = " ".join(df["return_reason"].dropna().astype(str)).strip()
    if not text:
        st.info("No return reason text available for word cloud.")
        return

    # Stop-word cleanup
    custom_stopwords = set(STOPWORDS).union(
        {
            "too",
            "my",
            "the",
            "from",
            "after",
            "one",
            "not",
            "and",
            "with",
            "for",
            "was",
            "were",
            "are",
            "item",
            "product",
            "order",
            "ordered",
            "because",
            "had",
            "have",
            "this",
            "that",
            "it",
        }
    )

    st.subheader("Return Reason Word Cloud")

    wordcloud = WordCloud(
        width=1000,
        height=450,
        background_color="white",
        collocations=False,
        stopwords=custom_stopwords,
    ).generate(text)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)


def render_overall_report(parsed):
    raw_category_counts = parsed.get("category_breakdown", {})

    category_counts = {}
    for k, v in raw_category_counts.items():
        try:
            category_counts[k] = int(v)
        except Exception:
            category_counts[k] = 0

    total_count = sum(category_counts.values()) if category_counts else 0

    st.subheader("Overall Return Analysis Report")

    if total_count > 0:
        top_category = max(category_counts, key=category_counts.get)
        top_count = category_counts[top_category]
        top_percentage = (top_count / total_count) * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Return Reasons", total_count)
        col2.metric("Top Issue", top_category)
        col3.metric("Top Issue %", f"{top_percentage:.1f}%")

        st.markdown("### Top Issues")
        for category, count in category_counts.items():
            percentage = (count / total_count) * 100
            st.write(f"- **{category}**: {count} cases ({percentage:.1f}%)")

        st.markdown("### Issue Category Chart")
        chart_df = pd.DataFrame.from_dict(
            category_counts, orient="index", columns=["count"]
        ).sort_values("count", ascending=False)
        st.bar_chart(chart_df)
    else:
        st.write("No category breakdown available.")

    st.markdown("### Top Phrases")
    top_phrases = parsed.get("top_phrases", [])
    if top_phrases:
        for phrase in top_phrases:
            st.write(f"- {phrase}")
    else:
        st.write("No recurring phrases identified.")

    st.markdown("### Recommended Fixes")
    suggestions = parsed.get("product_fix_suggestions", [])
    if suggestions:
        for i, suggestion in enumerate(suggestions, start=1):
            st.write(f"{i}. {suggestion}")
    else:
        st.write("No fix suggestions available.")

    st.markdown("### Estimated Return Reduction Opportunity")
    st.success(parsed.get("estimated_return_reduction_opportunity", "Not available"))

    with st.expander("Show Overall Raw JSON"):
        st.json(parsed)


def build_top_problem_products_table(df, parsed):
    if "product" not in df.columns:
        return None

    product_counts = df["product"].astype(str).value_counts().reset_index()
    product_counts.columns = ["Product", "Return Count"]

    product_insights = parsed.get("product_insights", [])
    if not product_insights:
        return None

    insight_rows = []
    for item in product_insights:
        insight_rows.append(
            {
                "Product": item.get("product", "Unknown Product"),
                "Top Issue": item.get("top_issue_category", "N/A"),
            }
        )

    insight_df = pd.DataFrame(insight_rows)
    if insight_df.empty:
        return None

    merged = product_counts.merge(insight_df, on="Product", how="left")
    merged["Top Issue"] = merged["Top Issue"].fillna("N/A")

    merged = merged[["Product", "Top Issue", "Return Count"]]
    merged = merged.sort_values(["Return Count", "Product"], ascending=[False, True]).reset_index(drop=True)
    return merged


def build_return_rate_table(df, orders_df):
    if "product" not in df.columns or orders_df is None:
        return None

    product_counts = df["product"].astype(str).value_counts().reset_index()
    product_counts.columns = ["Product", "Return Count"]

    merged_df = product_counts.merge(
        orders_df,
        how="left",
        left_on="Product",
        right_on="product",
    )

    merged_df["orders"] = pd.to_numeric(merged_df["orders"], errors="coerce")
    merged_df["Return Rate"] = merged_df["Return Count"] / merged_df["orders"]
    merged_df["Return Rate %"] = merged_df["Return Rate"] * 100

    return_rate_display = merged_df[["Product", "Return Count", "orders", "Return Rate %"]].copy()
    return_rate_display = return_rate_display.rename(columns={"orders": "Orders"})
    return_rate_display["Return Rate %"] = return_rate_display["Return Rate %"].round(2)

    return return_rate_display


def render_product_report(df, parsed, orders_df=None):
    st.subheader("Product Analysis Report")

    if "product" not in df.columns:
        st.warning("Product Analysis requires a `product` column in the returns CSV.")
        return

    product_counts = df["product"].astype(str).value_counts().reset_index()
    product_counts.columns = ["Product", "Return Count"]
    product_counts = product_counts.sort_values("Return Count", ascending=False)

    if not product_counts.empty:
        top_product = product_counts.iloc[0]["Product"]
        top_product_count = int(product_counts.iloc[0]["Return Count"])
        unique_products = int(product_counts["Product"].nunique())

        col1, col2, col3 = st.columns(3)
        col1.metric("Unique Products", unique_products)
        col2.metric("Most Returned Product", top_product)
        col3.metric("Top Product Return Count", top_product_count)

    st.markdown("### Product-Level Return Counts")
    st.dataframe(product_counts, use_container_width=True)

    product_counts_csv = product_counts.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Product Return Counts CSV",
        data=product_counts_csv,
        file_name="product_return_counts.csv",
        mime="text/csv",
    )

    st.markdown("### Product Return Chart")
    fig, ax = plt.subplots(figsize=(10, max(4, len(product_counts) * 0.6)))
    ax.barh(product_counts["Product"], product_counts["Return Count"])
    ax.invert_yaxis()
    ax.set_xlabel("Return Count")
    ax.set_ylabel("Product")
    ax.set_title("Returns by Product")
    st.pyplot(fig)

    top_problem_df = build_top_problem_products_table(df, parsed)
    if top_problem_df is not None and not top_problem_df.empty:
        st.markdown("### Top Problem Products")
        st.dataframe(top_problem_df, use_container_width=True)

        top_problem_csv = top_problem_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Top Problem Products CSV",
            data=top_problem_csv,
            file_name="top_problem_products.csv",
            mime="text/csv",
        )

    if orders_df is not None:
        st.markdown("### Return Rate by Product")
        return_rate_display = build_return_rate_table(df, orders_df)

        if return_rate_display is not None:
            st.dataframe(return_rate_display, use_container_width=True)

            return_rate_csv = return_rate_display.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Return Rate CSV",
                data=return_rate_csv,
                file_name="return_rate_by_product.csv",
                mime="text/csv",
            )

            valid_rate_df = return_rate_display.dropna(subset=["Orders", "Return Rate %"]).sort_values(
                "Return Rate %", ascending=False
            )

            if not valid_rate_df.empty:
                top_rate_product = valid_rate_df.iloc[0]["Product"]
                top_rate_value = float(valid_rate_df.iloc[0]["Return Rate %"])

                col1, col2 = st.columns(2)
                col1.metric("Highest Return Rate Product", top_rate_product)
                col2.metric("Highest Return Rate %", f"{top_rate_value:.2f}%")

                fig2, ax2 = plt.subplots(figsize=(10, max(4, len(valid_rate_df) * 0.6)))
                ax2.barh(valid_rate_df["Product"], valid_rate_df["Return Rate %"])
                ax2.invert_yaxis()
                ax2.set_xlabel("Return Rate %")
                ax2.set_ylabel("Product")
                ax2.set_title("Return Rate by Product")
                st.pyplot(fig2)
            else:
                st.info("No valid product/order matches found for return rate calculation.")

    product_insights = parsed.get("product_insights", [])

    st.markdown("### AI Product Insights")
    if product_insights:
        for item in product_insights:
            st.markdown(f"**{item.get('product', 'Unknown Product')}**")
            st.write(f"- Top issue: {item.get('top_issue_category', 'N/A')}")
            st.write(f"- Summary: {item.get('issue_summary', 'N/A')}")
            st.write(f"- Recommended fix: {item.get('recommended_fix', 'N/A')}")
    else:
        st.write("No AI product insights available.")

    with st.expander("Show Product Raw JSON"):
        st.json(parsed)


def build_sku_base_table(df, orders_df=None):
    if "product" not in df.columns:
        return pd.DataFrame()

    sku_df = (
        df.groupby("product")
        .agg(
            return_count=("return_reason", "count"),
            return_reasons=("return_reason", lambda x: list(x)),
        )
        .reset_index()
        .rename(columns={"product": "Product"})
    )

    if orders_df is not None and not orders_df.empty:
        orders_copy = orders_df.copy().rename(columns={"product": "Product", "orders": "Orders"})
        sku_df = sku_df.merge(orders_copy, on="Product", how="left")
        sku_df["Orders"] = pd.to_numeric(sku_df["Orders"], errors="coerce")
        sku_df["Return Rate %"] = (sku_df["return_count"] / sku_df["Orders"]) * 100
    else:
        sku_df["Orders"] = pd.NA
        sku_df["Return Rate %"] = pd.NA

    return sku_df


def diagnose_single_sku(client, product_name, reasons_list, return_count, orders_value, return_rate_value):
    reasons_text = "\n".join([str(r) for r in reasons_list])

    orders_text = "unknown" if pd.isna(orders_value) else str(int(orders_value))
    return_rate_text = "unknown" if pd.isna(return_rate_value) else f"{float(return_rate_value):.2f}"

    prompt = f"""
You are an e-commerce returns analyst.

Analyze the following SKU return data.

Product: {product_name}
Return count: {return_count}
Orders: {orders_text}
Return rate percent: {return_rate_text}

Return reasons:
{reasons_text}

Return valid JSON only.
Do not include markdown.
Do not include explanations.
Do not write anything before or after the JSON.

Rules:
- Identify the single most likely root cause
- Choose one top issue category from:
  - Sizing Issues
  - Quality Problems
  - Product Mismatch
  - Shipping Damage
  - Changed Mind
  - Other
- evidence_summary should briefly summarize the pattern in the return reasons
- recommended_fix should be specific and actionable
- confidence must be one of: High, Medium, Low
- priority must be one of: High, Medium, Low

Required JSON format:
{{
  "product": "",
  "top_issue_category": "",
  "likely_root_cause": "",
  "evidence_summary": "",
  "recommended_fix": "",
  "confidence": "",
  "priority": ""
}}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        response_format={"type": "json_object"},
    )

    result = response.choices[0].message.content
    parsed, error = clean_json_response(result)

    if parsed is None:
        raise ValueError(f"Invalid JSON for SKU {product_name}: {error}")

    return parsed


def build_sku_diagnosis_table(client, df, orders_df=None, min_returns=2, max_skus=10):
    sku_base_df = build_sku_base_table(df, orders_df=orders_df)
    if sku_base_df.empty:
        return pd.DataFrame()

    eligible_df = sku_base_df[sku_base_df["return_count"] >= min_returns].copy()

    eligible_df = eligible_df.sort_values(
        by=["return_count", "Return Rate %"],
        ascending=[False, False],
        na_position="last",
    ).head(max_skus)

    diagnosis_rows = []

    for _, row in eligible_df.iterrows():
        diagnosis = diagnose_single_sku(
            client=client,
            product_name=row["Product"],
            reasons_list=row["return_reasons"],
            return_count=int(row["return_count"]),
            orders_value=row["Orders"],
            return_rate_value=row["Return Rate %"],
        )

        diagnosis["Return Count"] = int(row["return_count"])
        diagnosis["Orders"] = None if pd.isna(row["Orders"]) else int(row["Orders"])
        diagnosis["Return Rate %"] = None if pd.isna(row["Return Rate %"]) else round(float(row["Return Rate %"]), 2)

        diagnosis_rows.append(diagnosis)

    if not diagnosis_rows:
        return pd.DataFrame()

    diagnosis_df = pd.DataFrame(diagnosis_rows)

    desired_columns = [
        "product",
        "top_issue_category",
        "likely_root_cause",
        "evidence_summary",
        "recommended_fix",
        "confidence",
        "priority",
        "Return Count",
        "Orders",
        "Return Rate %",
    ]

    for col in desired_columns:
        if col not in diagnosis_df.columns:
            diagnosis_df[col] = pd.NA

    diagnosis_df = diagnosis_df[desired_columns].rename(
        columns={
            "product": "Product",
            "top_issue_category": "Top Issue Category",
            "likely_root_cause": "Likely Root Cause",
            "evidence_summary": "Evidence Summary",
            "recommended_fix": "Recommended Fix",
            "confidence": "Confidence",
            "priority": "Priority",
        }
    )

    return diagnosis_df


def filter_diagnosis_by_priority(diagnosis_df, priority_filter):
    if diagnosis_df.empty:
        return diagnosis_df

    filtered_df = diagnosis_df.copy()

    if priority_filter == "High only":
        filtered_df = filtered_df[filtered_df["Priority"].astype(str).str.lower() == "high"]
    elif priority_filter == "Medium + High":
        filtered_df = filtered_df[
            filtered_df["Priority"].astype(str).str.lower().isin(["medium", "high"])
        ]

    return filtered_df.reset_index(drop=True)


def render_sku_diagnosis_section(diagnosis_df):
    st.subheader("SKU Diagnosis Table")

    if diagnosis_df.empty:
        st.info("No SKUs met the diagnosis threshold.")
        return

    priority_filter = st.selectbox(
        "Filter SKU diagnoses by priority",
        ["All", "High only", "Medium + High"],
        index=0,
    )

    filtered_df = filter_diagnosis_by_priority(diagnosis_df, priority_filter)

    if filtered_df.empty:
        st.warning("No SKU diagnoses match the selected priority filter.")
        return

    st.dataframe(filtered_df, use_container_width=True)

    csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Filtered SKU Diagnosis CSV",
        data=csv_bytes,
        file_name="sku_diagnosis_filtered.csv",
        mime="text/csv",
    )

    st.markdown("### SKU Diagnosis Details")
    for _, row in filtered_df.iterrows():
        with st.expander(f"{row['Product']} — {row['Top Issue Category']} — Priority: {row['Priority']}"):
            st.write(f"**Return Count:** {row['Return Count']}")
            if pd.notna(row["Orders"]):
                st.write(f"**Orders:** {row['Orders']}")
            if pd.notna(row["Return Rate %"]):
                st.write(f"**Return Rate %:** {row['Return Rate %']}")
            st.write(f"**Likely Root Cause:** {row['Likely Root Cause']}")
            st.write(f"**Evidence Summary:** {row['Evidence Summary']}")
            st.write(f"**Recommended Fix:** {row['Recommended Fix']}")
            st.write(f"**Confidence:** {row['Confidence']}")
            st.write(f"**Priority:** {row['Priority']}")


# -----------------------------
# Main
# -----------------------------
if returns_file:
    try:
        raw_df = pd.read_csv(returns_file)
    except Exception as e:
        st.error(f"Could not read returns CSV file: {str(e)}")
        st.stop()

    if "return_reason" not in raw_df.columns:
        st.error("Returns CSV must contain a `return_reason` column.")
        st.stop()

    st.subheader("Raw Returns Data Preview")
    st.dataframe(raw_df.head(), use_container_width=True)

    df, quality_summary = run_data_quality_checks(raw_df)
    render_data_quality_summary(quality_summary)

    orders_df = None
    if orders_file is not None:
        orders_df = load_and_clean_orders(orders_file)
        if orders_df is not None:
            st.subheader("Orders Data Preview")
            st.dataframe(orders_df.head(), use_container_width=True)

    has_date = "date" in df.columns

    st.subheader("Cleaned Returns Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    if has_date:
        valid_dates = df["date"].dropna()
        if not valid_dates.empty:
            min_date = valid_dates.min().date()
            max_date = valid_dates.max().date()

            st.markdown("### Date Filter")
            date_range = st.date_input(
                "Select date range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
            )

            if isinstance(date_range, tuple) and len(date_range) == 2:
                start_date, end_date = date_range
                df = df[
                    (df["date"].dt.date >= start_date)
                    & (df["date"].dt.date <= end_date)
                ]
                st.write(f"Filtered date range: **{start_date}** to **{end_date}**")
        else:
            st.warning("A `date` column exists, but no valid dates were found.")

    total_rows = len(df)
    st.write(f"Total records available for analysis: **{total_rows}**")

    if total_rows == 0:
        st.error("No usable return reasons found after cleaning/filtering.")
        st.stop()

    if analysis_mode == "Product Analysis" and "product" not in df.columns:
        st.warning("You selected Product Analysis, but your returns CSV does not include a `product` column.")

    if st.button("Analyze Returns"):
        with st.spinner("Analyzing returns..."):
            try:
                overall_parsed = None
                product_parsed = None

                if analysis_mode in ["Overall Analysis", "Both"]:
                    overall_text = "\n".join(df["return_reason"].tolist())

                    overall_prompt = f"""
You are an e-commerce returns analyst.

Analyze the following return reasons:

{overall_text}

Return valid JSON only.
Do not include any explanation.
Do not include markdown.
Do not use code fences.
Do not write anything before or after the JSON.

Rules:
- category_breakdown must be counts, not percentages
- Use only these categories:
  - Sizing Issues
  - Quality Problems
  - Product Mismatch
  - Shipping Damage
  - Changed Mind
  - Other
- top_phrases should be a short list of recurring phrases
- product_fix_suggestions should be specific and actionable
- estimated_return_reduction_opportunity should be a short range like "10-15%"

Required JSON format:
{{
  "category_breakdown": {{
    "Sizing Issues": 0,
    "Quality Problems": 0,
    "Product Mismatch": 0,
    "Shipping Damage": 0,
    "Changed Mind": 0,
    "Other": 0
  }},
  "top_phrases": [],
  "product_fix_suggestions": [],
  "estimated_return_reduction_opportunity": ""
}}
"""

                    overall_response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": overall_prompt}],
                        temperature=0.2,
                        response_format={"type": "json_object"},
                    )

                    overall_result = overall_response.choices[0].message.content
                    overall_parsed, overall_error = clean_json_response(overall_result)

                    if overall_parsed is None:
                        st.error("The model did not return valid JSON for overall analysis.")
                        st.write("Raw model output:")
                        st.code(str(overall_error))
                        st.stop()

                if analysis_mode in ["Product Analysis", "Both"] and "product" in df.columns:
                    product_text = "\n".join(
                        df.apply(
                            lambda row: f"Product: {row['product']} | Return Reason: {row['return_reason']}",
                            axis=1,
                        ).tolist()
                    )

                    product_prompt = f"""
You are an e-commerce returns analyst.

Analyze the following product-level return records:

{product_text}

Return valid JSON only.
Do not include any explanation.
Do not include markdown.
Do not use code fences.
Do not write anything before or after the JSON.

Rules:
- Focus on product-specific issue patterns
- product_insights should identify the top issue category for each product
- issue_summary should briefly explain the recurring problem for that product
- recommended_fix should be specific and actionable for that product

Required JSON format:
{{
  "product_insights": [
    {{
      "product": "",
      "top_issue_category": "",
      "issue_summary": "",
      "recommended_fix": ""
    }}
  ]
}}
"""

                    product_response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": product_prompt}],
                        temperature=0.2,
                        response_format={"type": "json_object"},
                    )

                    product_result = product_response.choices[0].message.content
                    product_parsed, product_error = clean_json_response(product_result)

                    if product_parsed is None:
                        st.error("The model did not return valid JSON for product analysis.")
                        st.write("Raw model output:")
                        st.code(str(product_error))
                        st.stop()

                render_trend_section(df)
                render_word_cloud(df)

                if overall_parsed:
                    render_overall_report(overall_parsed)

                if product_parsed:
                    render_product_report(df, product_parsed, orders_df=orders_df)

                if "product" in df.columns:
                    with st.spinner("Building SKU diagnosis table..."):
                        diagnosis_df = build_sku_diagnosis_table(
                            client=client,
                            df=df,
                            orders_df=orders_df,
                            min_returns=min_returns_per_sku,
                            max_skus=max_skus_to_diagnose,
                        )
                    render_sku_diagnosis_section(diagnosis_df)

            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")