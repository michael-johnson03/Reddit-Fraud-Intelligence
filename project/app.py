import os
# Working directory should contain Reddit pipeline outputs and Master_File_2020_Filtered.csv.
# In Colab the team workflow used os.chdir("/content"); locally, run this file from a directory
# that contains the required data files (see README for required files).

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import re
import glob
import json
import numpy as np
from datetime import datetime

# ── 1. PAGE CONFIG & STYLES ───────────────────────────────────────────────
st.set_page_config(page_title="Executive Fraud Intelligence", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 3.5rem !important; padding-bottom: 2rem; padding-left: 3rem; padding-right: 3rem; }
    .centered-title { text-align: center; width: 100%; color: #003366; font-size: 1.8rem; font-weight: 800; margin-bottom: 0.2rem; }
    span[data-baseweb="tag"] { display: none !important; }
    div[data-baseweb="select"] > div:first-child { height: 42px !important; }
    label { font-size: 0.85rem !important; font-weight: 600; color: #444; }
    h3 { font-size: 1.1rem !important; font-weight: bold !important; color: #1f2937; border-bottom: 1px solid #eee; margin-bottom: 5px !important; }
    .chart-subtitle { font-size: 0.8rem; color: #777; margin-bottom: 10px; font-style: italic; }
    div[data-testid="stMetric"] {
        background-color: #f0f2f6;
        padding: 6px 10px;
        border-radius: 8px;
        border-left: 4px solid #003366;
    }
    div[data-testid="stMetric"] label { font-size: 0.72rem !important; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { font-size: 1rem !important; }
    .summary-section { margin-bottom: 10px; }
    .summary-label { font-weight: 700; color: #003366; font-size: 13px; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 3px; }
    .summary-text { color: #1a1a2e; font-size: 14px; line-height: 1.5; }
    .watch-pill { display: inline-block; background-color: #003366; color: white; border-radius: 20px; padding: 2px 10px; font-size: 11px; margin: 2px 3px 2px 0; }
</style>
""", unsafe_allow_html=True)


# ── 2. DATA LOADERS ───────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('Master_File_2020_Filtered.csv')
    df['date'] = pd.to_datetime(df['date'])
    if df['date'].dt.tz is not None:
        df['date'] = df['date'].dt.tz_localize(None)
    df = df[df['date'] <= datetime.now()]
    df['primary_tag'] = df['primary_tag'].fillna('Uncategorized').str.title().str.strip()
    df['primary_tag_norm'] = df['primary_tag'].str.replace(' ', '_')
    df['source'] = df['source'].str.upper().str.strip()
    return df


@st.cache_data
def load_reddit_posts():
    if os.path.exists("reddit_posts_all.parquet"):
        df = pd.read_parquet("reddit_posts_all.parquet")
    else:
        files = sorted(glob.glob("reddit_posts_themed*.parquet"))
        if not files:
            return pd.DataFrame()
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        df = df.drop_duplicates(subset=["post_id"], keep="first")
    df['run_date'] = pd.to_datetime(df['run_date'])
    if 'published_at' in df.columns:
        df['post_date'] = pd.to_datetime(df['published_at'], errors='coerce', utc=True)
        df['post_date'] = df['post_date'].dt.tz_localize(None)
    else:
        df['post_date'] = df['run_date']
    df['theme'] = df['theme'].fillna('Other_Unclear').astype(str)
    df['rss_summary_text'] = df['rss_summary_text'].fillna('').astype(str)
    df['title'] = df['title'].fillna('').astype(str)
    df['post_id'] = df['post_id'].fillna('').astype(str)
    return df


@st.cache_data
def load_reddit_summaries():
    if not os.path.exists("reddit_theme_summaries.parquet"):
        return pd.DataFrame()
    df = pd.read_parquet("reddit_theme_summaries.parquet")
    df['run_date'] = pd.to_datetime(df['run_date'])
    df['theme'] = df['theme'].fillna('Other_Unclear').astype(str)
    df['theme_summary_text'] = df['theme_summary_text'].fillna('').astype(str)
    return df


df               = load_data()
reddit_posts     = load_reddit_posts()
reddit_summaries = load_reddit_summaries()

# ── 3. HEADER & TABS ─────────────────────────────────────────────────────
st.markdown('<div class="centered-title">STRATEGIC FRAUD INTELLIGENCE & VELOCITY ANALYSIS</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Executive Dashboard",
    "🔍 Source Audit",
    "🌐 Reddit Intelligence",
    "🤖 Cross Intelligence"
])


# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — EXECUTIVE DASHBOARD
# ══════════════════════════════════════════════════════════════════════════
with tab1:
    row1_col1, row1_col2 = st.columns([1, 1])
    row2_col1, row2_col2, row2_col3, row2_col4 = st.columns([0.7, 1, 1, 1])

    with row2_col1:
        st.subheader("🛡️ Strategic Filters")
        timeframe = st.selectbox("Interval", ["Weekly", "Monthly", "Quarterly", "Yearly"], index=1)
        all_tags = sorted(df['primary_tag'].unique())
        default_tags = [t for t in all_tags if t.lower() != 'other']
        selected_sources = st.multiselect("Sources", sorted(df['source'].unique()), default=list(df['source'].unique()))
        selected_tags = st.multiselect("Categories", all_tags, default=default_tags)

    tf_map     = {"Weekly": "W-SUN", "Monthly": "ME", "Quarterly": "QE", "Yearly": "YE"}
    window_map = {"Weekly": 10, "Monthly": 12, "Quarterly": 8, "Yearly": 999}
    label_map  = {"Weekly": "Last 10 Weeks", "Monthly": "Last 12 Months", "Quarterly": "Last 8 Quarters", "Yearly": "Historical Overview"}

    mask = (df['source'].isin(selected_sources)) & (df['primary_tag'].isin(selected_tags))
    full_filtered = df[mask].copy()

    if not full_filtered.empty:
        temp_grouped = full_filtered.groupby(pd.Grouper(key='date', freq=tf_map[timeframe])).size()
        valid_periods = temp_grouped[temp_grouped.index <= datetime.now()].tail(window_map[timeframe]).index
        filtered_df = full_filtered[(full_filtered['date'] >= valid_periods[0]) & (full_filtered['date'] <= valid_periods[-1])]
    else:
        filtered_df = full_filtered

    with row1_col1:
        st.subheader(f"Historical Intelligence Trend Analysis ({timeframe})")
        st.markdown(f'<p class="chart-subtitle">Analyzing volume trends for the {label_map[timeframe]}</p>', unsafe_allow_html=True)
        if not filtered_df.empty:
            v_data = filtered_df.groupby([pd.Grouper(key='date', freq=tf_map[timeframe]), 'primary_tag']).size().reset_index(name='count')
            fig_v = px.line(v_data, x='date', y='count', color='primary_tag', markers=True, height=300, template="plotly_white")
            fig_v.update_layout(margin=dict(l=0, r=0, t=30, b=0), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10), title=None))
            st.plotly_chart(fig_v, use_container_width=True)

    with row1_col2:
        st.subheader(f"({timeframe}) Percent Change in Fraud Categories")
        pivot = full_filtered.groupby([pd.Grouper(key='date', freq=tf_map[timeframe]), 'primary_tag']).size().unstack(fill_value=0)
        pivot = pivot[pivot.sum(axis=1) > 0]

        if len(pivot) > 1:
            if timeframe == "Yearly":
                latest_date = full_filtered['date'].max()
                day_of_year = latest_date.timetuple().tm_yday
                curr_data = full_filtered[full_filtered['date'].dt.year == latest_date.year]
                prev_data = full_filtered[(full_filtered['date'].dt.year == latest_date.year - 1) & (full_filtered['date'].dt.dayofyear <= day_of_year)]
                curr_vals = curr_data.groupby('primary_tag').size()
                prev_vals = prev_data.groupby('primary_tag').size().reindex(curr_vals.index, fill_value=0)
                label_text = f"YTD: Jan-{latest_date.strftime('%b')} {latest_date.year-1} vs {latest_date.year}"
            elif timeframe == "Quarterly":
                idx, prev_idx = (-2, -3) if len(pivot) >= 3 else (-1, -2)
                curr_vals, prev_vals = pivot.iloc[idx], pivot.iloc[prev_idx]
                c_p, p_p = pivot.index[idx], pivot.index[prev_idx]
                label_text = f"Full Qtr: Q{(p_p.month-1)//3+1} {p_p.year} vs Q{(c_p.month-1)//3+1} {c_p.year}"
            elif timeframe == "Monthly":
                idx, prev_idx = (-2, -3) if len(pivot) >= 3 else (-1, -2)
                curr_vals, prev_vals = pivot.iloc[idx], pivot.iloc[prev_idx]
                label_text = f"Monthly: {pivot.index[prev_idx].strftime('%b %y')} vs {pivot.index[idx].strftime('%b %y')}"
            else:
                idx, prev_idx = (-2, -3) if len(pivot) >= 3 else (-1, -2)
                curr_vals, prev_vals = pivot.iloc[idx], pivot.iloc[prev_idx]
                c_start, p_start = pivot.index[idx], pivot.index[prev_idx]
                label_text = f"Weekly: {p_start.strftime('%b %d')} vs {c_start.strftime('%b %d')}"

            diff_df = (curr_vals - prev_vals).reset_index(name='Change')
            diff_df['%'] = (diff_df['Change'] / prev_vals.replace(0, 1).values * 100)
            st.markdown(f"**{label_text}**")
            fig_a = px.bar(diff_df, x='primary_tag', y='Change', color='Change',
                           text=diff_df['%'].apply(lambda x: f"{x:+.0f}%"),
                           height=350, template="plotly_white", color_continuous_scale='RdBu_r')
            fig_a.update_traces(textposition='outside', cliponaxis=False)
            y_max, y_min = diff_df['Change'].max(), diff_df['Change'].min()
            fig_a.update_yaxes(range=[y_min * 1.4 if y_min < 0 else -2, y_max * 1.4 if y_max > 0 else 2])
            fig_a.update_layout(margin=dict(l=10, r=10, t=30, b=80), coloraxis_showscale=False, xaxis_title=None, yaxis_title="Delta (Article Count)")
            fig_a.update_xaxes(tickangle=45)
            st.plotly_chart(fig_a, use_container_width=True)
        else:
            st.info("Awaiting more historical data to calculate deltas.")

    with row2_col2:
        st.subheader("🌐 Source Distribution")
        if not filtered_df.empty:
            source_counts = filtered_df['source'].value_counts().reset_index()
            source_counts.columns = ['Source', 'Count']
            fig_pie = px.pie(source_counts, values='Count', names='Source', hole=0.3, color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_pie.update_traces(textposition='auto', textinfo='percent+label', textfont_size=9, insidetextorientation='horizontal')
            fig_pie.update_layout(margin=dict(l=10, r=10, t=5, b=5), height=220, showlegend=False)
            st.plotly_chart(fig_pie, use_container_width=True)

    with row2_col3:
        st.subheader("Top 10 Fraud Types")
        top_tags = filtered_df['primary_tag'].value_counts().head(10).reset_index()
        fig_b = px.bar(top_tags, x='count', y='primary_tag', orientation='h', height=260, color_discrete_sequence=['#003366'])
        fig_b.update_layout(margin=dict(l=0, r=0, t=0, b=0), yaxis={'categoryorder': 'total ascending'}, xaxis_title=None, yaxis_title=None)
        st.plotly_chart(fig_b, use_container_width=True)

    with row2_col4:
        st.subheader("Narrative Themes")
        final_stops = set(STOPWORDS)
        noise_words = ["say","says","said","term","terms","according","will","also","new","used","use","using",
                       "one","many","across","including","within","without","provide","report","article",
                       "scammer","scammers","scams","fraud","payment","payments","continue","tool",
                       "identity","platform","platforms","companies","help","information","system","systems",
                       "account","bank","form","data","member","customer","customers","users","user","company",
                       "consumer","consumers","attack","attacks","transaction","transactions","services",
                       "security","access","activity","working","time","need","make","even","first","way",
                       "people","often","pymnts","financial","victim","fraudsters","banks","sharing","real",
                       "fbi","schemes","alert","treasury","us","employee","news","attacker","institution",
                       "device","trust","risk","today","day","ftc","threat","year","victims","find","fake",
                       "what'","may","found","scam","bleeping computer","see"]
        final_stops.update([w.lower() for w in noise_words])
        raw_text = " ".join(filtered_df['body'].dropna().astype(str))
        cleaned = re.sub(r"\b[sS]\b", "", raw_text)
        cleaned = re.sub(r"'s\b|'S\b", "", cleaned).lower()
        if len(cleaned) > 20:
            wc = WordCloud(stopwords=final_stops, width=400, height=260, background_color='white', colormap='Dark2', max_words=50, collocations=False).generate(cleaned)
            fig_wc, ax = plt.subplots(figsize=(4, 2.6))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig_wc)


# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — SOURCE AUDIT
# ══════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 🔍 Source Integrity & Audit Report")
    st.markdown("#### **Category Distribution by Source**")
    bias_df = df.groupby(['primary_tag', 'source']).size().unstack(fill_value=0)
    bias_df['Total'] = bias_df.sum(axis=1)
    st.dataframe(bias_df.sort_values('Total', ascending=False), use_container_width=True)
    st.divider()

    col_audit, col_metric = st.columns([0.7, 0.3])
    with col_audit:
        st.markdown("#### **Intelligence Source Audit**")
        audit_data = []
        for source in df['source'].unique():
            s_data = df[df['source'] == source]
            audit_data.append({
                "Source": source.upper(),
                "Article Count": len(s_data),
                "First Seen": s_data['date'].min().strftime('%Y-%m-%d'),
                "Last Seen": s_data['date'].max().strftime('%Y-%m-%d')
            })
        st.table(pd.DataFrame(audit_data).sort_values('Article Count', ascending=False))

    with col_metric:
        st.markdown("#### **Summary**")
        try:
            with st.container():
                st.metric(label="Total Unique Leads", value=f"{len(df):,}")
                st.info(f"Analysis covers **{len(df['source'].unique())}** verified intelligence vendors.")
        except Exception:
            st.write(f"**Total Unique Leads:** {len(df):,}")
            st.write(f"**Verified Vendors:** {len(df['source'].unique())}")


# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — REDDIT INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════
with tab3:

    if reddit_posts.empty or reddit_summaries.empty:
        st.warning("Reddit data files not found. Copy reddit_posts_all.parquet and reddit_theme_summaries.parquet alongside app.py and relaunch.")
        st.stop()

    # Reddit logo + title
    st.markdown('''<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
        <img src="https://www.redditstatic.com/desktop2x/img/favicon/android-icon-192x192.png" width="28" style="border-radius:50%;">
        <span style="font-size:1.2rem;font-weight:800;color:#003366;">Reddit Fraud Intelligence</span>
    </div>''', unsafe_allow_html=True)

    # Controls + metrics in one row
    ctrl_col, m1_col, m2_col, m3_col, m4_col = st.columns([1.5, 1, 1, 1.2, 1.2])
    with ctrl_col:
        all_themes = ["All Themes"] + sorted(reddit_posts['theme'].unique())
        sel_theme  = st.selectbox("Fraud Category", all_themes, index=0, key="reddit_theme")

    # Filter data
    if sel_theme == "All Themes":
        theme_posts   = reddit_posts.copy()
        theme_summary = reddit_summaries.copy()
    else:
        theme_posts   = reddit_posts[reddit_posts['theme'] == sel_theme].copy()
        theme_summary = reddit_summaries[reddit_summaries['theme'] == sel_theme].copy()

    run_dates   = sorted(theme_posts['run_date'].dt.date.unique())
    total_posts = len(theme_posts)
    n_runs      = len(run_dates)

    date_range = "—"
    if 'post_date' in theme_posts.columns:
        try:
            valid = theme_posts['post_date'].dropna()
            if len(valid) > 0:
                date_range = f"{valid.min().date()} → {valid.max().date()}"
        except Exception:
            pass

    if n_runs >= 2:
        count_by_run = theme_posts.groupby('run_date').size().sort_index()
        last_two = count_by_run.iloc[-2:]
        pct_chg  = ((last_two.iloc[-1] - last_two.iloc[-2]) / last_two.iloc[-2] * 100) if last_two.iloc[-2] != 0 else 0.0
        pct_label = f"{pct_chg:+.1f}% vs prior run"
    else:
        pct_chg, pct_label = 0.0, "Only one run available"

    with m1_col:
        st.metric("Total Posts", f"{total_posts:,}")
    with m2_col:
        st.metric("Pipeline Runs", f"{n_runs}")
    with m3_col:
        st.metric("Date Range", date_range)
    with m4_col:
        st.metric("Vol Change", pct_label, delta=f"{pct_chg:+.1f}%" if n_runs >= 2 else None)

    # Summary card
    def parse_summary(text):
        pattern  = re.compile(r"(Pattern:|Targets & Impact:|Watch signals:)", re.IGNORECASE)
        parts    = pattern.split(text)
        sections = {}
        for i in range(1, len(parts) - 1, 2):
            key = parts[i].rstrip(":").strip()
            val = parts[i + 1].strip()
            sections[key] = val
        return sections

    latest_summary_row = theme_summary.sort_values("run_date").iloc[-1] if not theme_summary.empty else None

    with st.container(border=True):
        st.subheader("📋 Theme Intelligence Summary")
        if sel_theme == "All Themes":
            st.info("👈 Select a specific fraud category above to view its intelligence summary.")
        elif latest_summary_row is not None:
            sections = parse_summary(latest_summary_row['theme_summary_text'])
            if "Pattern" in sections:
                st.markdown(f"<div class='summary-section'><div class='summary-label'>Pattern</div><div class='summary-text'>{sections['Pattern']}</div></div>", unsafe_allow_html=True)
            if "Targets & Impact" in sections:
                st.markdown(f"<div class='summary-section'><div class='summary-label'>Targets & Impact</div><div class='summary-text'>{sections['Targets & Impact']}</div></div>", unsafe_allow_html=True)
            if "Watch signals" in sections:
                signals = [s.strip() for s in sections['Watch signals'].split(',') if s.strip()]
                pills   = " ".join([f"<span class='watch-pill'>{s}</span>" for s in signals])
                st.markdown(f"<div class='summary-section'><div class='summary-label'>Watch Signals</div><div>{pills}</div></div>", unsafe_allow_html=True)
            st.caption(f"Generated {latest_summary_row['run_date'].date()} · Model: {latest_summary_row.get('model_name', '—')}")
        else:
            st.info("No summary available for this theme yet.")

    # Charts
    col_left, col_right = st.columns([2, 1])

    with col_left:
        with st.container(border=True):
            st.subheader("⚠️ Volume Change by Fraud Theme — Latest vs Prior Run")
            if n_runs >= 2:
                theme_run_counts = theme_posts.groupby(['run_date', 'theme']).size().unstack(fill_value=0).sort_index()
                if len(theme_run_counts) >= 2:
                    last_run  = theme_run_counts.iloc[-1]
                    prior_run = theme_run_counts.iloc[-2]
                    pct_by_theme = ((last_run - prior_run) / prior_run.replace(0, 1) * 100).sort_values(ascending=False)
                    chg_df = pct_by_theme.reset_index()
                    chg_df.columns = ['theme', 'pct_change']
                    chg_df = chg_df[chg_df['pct_change'] != 0]
                    chg_df['theme_label'] = chg_df['theme'].str.replace('_', ' ')
                    chg_df['color'] = chg_df['pct_change'].apply(lambda x: '#2ecc71' if x > 0 else '#e74c3c')
                    chg_df['indicator'] = chg_df['pct_change'].apply(lambda x: f"▲ {x:.1f}%" if x > 0 else f"▼ {abs(x):.1f}%")
                    if not chg_df.empty:
                        fig_chg = go.Figure(go.Bar(x=chg_df['pct_change'], y=chg_df['theme_label'], orientation='h',
                                                   marker_color=chg_df['color'].tolist(), text=chg_df['indicator'], textposition='outside',
                                                   hovertemplate='%{y}<br>Change: %{x:.1f}%<extra></extra>'))
                        fig_chg.update_layout(template='plotly_white', height=max(180, len(chg_df) * 22),
                                             xaxis_title="% Change", yaxis=dict(categoryorder='total ascending'),
                                             margin=dict(t=10, b=10, l=180))
                        st.plotly_chart(fig_chg, use_container_width=True)
                        top, bot = chg_df.iloc[0], chg_df.iloc[-1]
                        mag = "sharply" if abs(top['pct_change']) > 50 else ("moderately" if abs(top['pct_change']) > 20 else "slightly")
                        st.caption(f"📌 **{top['theme_label']}** {mag} {'increased' if top['pct_change'] > 0 else 'decreased'} by **{abs(top['pct_change']):.1f}%**. **{bot['theme_label']}** showed the largest decline at **{abs(bot['pct_change']):.1f}%**.")
            else:
                st.info("Need at least two pipeline runs to show % change.")

        with st.container(border=True):
            st.subheader("📈 Post Volume by Pipeline Run")
            tl_data = theme_posts.groupby('run_date').size().reset_index(name='posts').sort_values('run_date').reset_index(drop=True)
            if not tl_data.empty:
                tl_data = tl_data[tl_data['run_date'] <= tl_data['run_date'].max()]
                tl_data['change'] = tl_data['posts'].diff()
                tl_data['indicator'] = tl_data['change'].apply(lambda x: "▲" if x > 0 else ("▼" if x < 0 else "—") if pd.notna(x) else "—")
                tl_data['color'] = tl_data['change'].apply(lambda x: "#2ecc71" if x > 0 else ("#e74c3c" if x < 0 else "#95a5a6") if pd.notna(x) else "#95a5a6")
                fig_vel = go.Figure()
                fig_vel.add_trace(go.Scatter(x=tl_data['run_date'], y=tl_data['posts'], mode='lines+markers+text',
                                             text=tl_data['indicator'], textposition='top center', textfont=dict(size=14),
                                             marker=dict(size=10, color=tl_data['color'].tolist()),
                                             line=dict(color='#003366', width=2),
                                             hovertemplate='%{x|%Y-%m-%d}<br>Posts: %{y}<extra></extra>'))
                fig_vel.update_layout(template='plotly_white', height=180,
                                      xaxis=dict(title="Run Date", range=[tl_data['run_date'].min() - pd.Timedelta(days=3), tl_data['run_date'].max() + pd.Timedelta(days=3)]),
                                      yaxis_title="Post Count", margin=dict(t=5, b=5))
                st.plotly_chart(fig_vel, use_container_width=True)

        with st.container(border=True):
            st.subheader("🗓️ Pipeline Run Timeline")
            if not tl_data.empty:
                label = sel_theme.replace('_', ' ') if sel_theme != "All Themes" else "All Themes"
                fig_tl = go.Figure()
                fig_tl.add_trace(go.Scatter(x=tl_data['run_date'], y=[label] * len(tl_data), mode='markers+text',
                                            marker=dict(size=tl_data['posts'].apply(lambda x: max(10, min(40, x // 3))), color='#003366', opacity=0.7),
                                            text=tl_data['posts'].astype(str) + ' posts', textposition='top center',
                                            hovertemplate='%{x|%Y-%m-%d}<br>Posts: %{text}<extra></extra>'))
                fig_tl.update_layout(template='plotly_white', height=110,
                                     xaxis=dict(title="Run Date", range=[tl_data['run_date'].min() - pd.Timedelta(days=3), tl_data['run_date'].max() + pd.Timedelta(days=3)]),
                                     yaxis=dict(showticklabels=False), margin=dict(t=5, b=5))
                st.plotly_chart(fig_tl, use_container_width=True)
                st.caption("Bubble size = post volume on that run date")

    with col_right:
        with st.container(border=True):
            st.subheader("🏆 Top 10 Fraud Themes")
            theme_counts = reddit_posts['theme'].value_counts().head(10).reset_index()
            theme_counts.columns = ['Theme', 'Total']
            theme_counts['Theme'] = theme_counts['Theme'].str.replace('_', ' ')
            fig_top = px.bar(theme_counts, x='Total', y='Theme', orientation='h', color_discrete_sequence=['#003366'])
            fig_top.update_layout(yaxis={'categoryorder': 'total ascending'}, height=200, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig_top, use_container_width=True)

        with st.container(border=True):
            st.subheader("🔍 Signal Word Cloud")
            signal_text = re.sub(r'[,;|]', ' ', " ".join(theme_posts['theme_matches'].fillna('').astype(str).tolist())).strip()
            if len(signal_text.strip()) > 20:
                wc = WordCloud(width=500, height=200, background_color='white', colormap='Blues', max_words=60, collocations=False).generate(signal_text)
                fig_wc, ax = plt.subplots(figsize=(5, 2.2))
                ax.imshow(wc, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig_wc)
                st.caption("Fraud taxonomy signal matches only")
            else:
                st.info("Not enough signal data to generate word cloud.")

        with st.container(border=True):
            st.subheader("📰 Recent Posts")
            recent = theme_posts.sort_values('post_date', ascending=False).head(5).copy()
            recent['Post Date'] = recent['post_date'].dt.date.astype(str)
            recent['Link'] = recent['post_id'].apply(lambda x: x if x.startswith('http') else '')
            st.dataframe(recent[['Post Date', 'title', 'Link']], use_container_width=True, hide_index=True,
                         column_config={"Link": st.column_config.LinkColumn("Link"),
                                        "Post Date": st.column_config.TextColumn("Date", width="small"),
                                        "title": st.column_config.TextColumn("Title", width="large")})


# ══════════════════════════════════════════════════════════════════════════
# TAB 4 — CROSS INTELLIGENCE & FRAUD CHATBOT
# ══════════════════════════════════════════════════════════════════════════
with tab4:

    st.markdown('''<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
        <span style="font-size:1.2rem;font-weight:800;color:#003366;">🤖 Cross-Source Intelligence & Fraud Chatbot</span>
    </div>''', unsafe_allow_html=True)
    st.markdown('<p style="color:#666;font-size:0.85rem;margin-bottom:12px;">Compare institutional sources against Reddit signal. Ask questions grounded in both datasets.</p>', unsafe_allow_html=True)

    if df.empty or reddit_posts.empty:
        st.warning("Both master data and Reddit data are required for cross-analysis.")
        st.stop()

    # Theme map — Reddit themes → institutional category names (normalized with underscores)
    THEME_MAP = {
        "ATO_Account_Takeover":           "Identity_Fraud",
        "Phishing_Smishing":              "Cybercrime",
        "Identity_Theft":                 "Identity_Fraud",
        "Benefits_Fraud":                 "Consumer_Fraud",
        "Payment_Scams_P2P":              "Consumer_Fraud",
        "Check_Fraud":                    "Consumer_Fraud",
        "Crypto_Fraud":                   "Crypto_Fraud",
        "Tech_Support_Scam":              "Cybercrime",
        "Investment_Scam":                "Investment_Fraud",
        "BEC_Business_Email_Compromise":  "Cybercrime",
        "Money_Laundering":               "Money_Laundering",
        "Sanctions":                      "Money_Laundering",
        "Terrorist_Financing":            "Money_Laundering",
        "Human_Trafficking":              "Human_Trafficking",
        "Elder_Fraud":                    "Consumer_Fraud",
        "Military_Scam":                  "Consumer_Fraud",
        "Data_Breach":                    "Cybercrime",
        "Consumer_Billing_Fraud":         "Consumer_Fraud",
        "General_Scam":                   "Consumer_Fraud",
        "Other_Unclear":                  "Other",
    }

    # ── SECTION 1: COMPARISON CHART ───────────────────────────────────────
    st.subheader("📊 Institutional vs Reddit Signal Comparison")
    comp_col1, comp_col2 = st.columns([1, 1])

    with comp_col1:
        inst_counts = df['primary_tag_norm'].value_counts().reset_index()
        inst_counts.columns = ['Category', 'Institutional']

        reddit_mapped = reddit_posts.copy()
        reddit_mapped['mapped_category'] = reddit_mapped['theme'].map(THEME_MAP).fillna('Other')
        reddit_counts = reddit_mapped['mapped_category'].value_counts().reset_index()
        reddit_counts.columns = ['Category', 'Reddit']

        combined = inst_counts.merge(reddit_counts, on='Category', how='outer').fillna(0)
        combined['Institutional'] = combined['Institutional'].astype(int)
        combined['Reddit']        = combined['Reddit'].astype(int)
        combined['Inst_%']   = (combined['Institutional'] / combined['Institutional'].sum() * 100).round(1)
        combined['Reddit_%'] = (combined['Reddit'] / combined['Reddit'].sum() * 100).round(1)
        combined['Gap']      = (combined['Reddit_%'] - combined['Inst_%']).round(1)
        combined = combined.sort_values('Gap', ascending=False)

        fig_gap = go.Figure()
        fig_gap.add_trace(go.Bar(name='Institutional %', x=combined['Category'].str.replace('_', ' '), y=combined['Inst_%'], marker_color='#003366', opacity=0.8))
        fig_gap.add_trace(go.Bar(name='Reddit %',        x=combined['Category'].str.replace('_', ' '), y=combined['Reddit_%'], marker_color='#FF4500', opacity=0.8))
        fig_gap.update_layout(barmode='group', template='plotly_white', height=320,
                              legend=dict(orientation="h", y=1.1), xaxis_tickangle=35,
                              margin=dict(t=10, b=80, l=0, r=0), yaxis_title="% of Total Volume")
        st.plotly_chart(fig_gap, use_container_width=True)
        st.caption("Categories where Reddit % > Institutional % may be **leading indicators** not yet captured by formal sources.")

    with comp_col2:
        st.markdown("**Signal Gap Analysis**")
        st.caption("Positive gap = Reddit ahead of institutional sources (early warning). Negative = institutional leading.")
        display_df = combined[['Category', 'Institutional', 'Reddit', 'Inst_%', 'Reddit_%', 'Gap']].copy()
        display_df['Category'] = display_df['Category'].str.replace('_', ' ')
        display_df.columns = ['Category', 'Inst. Docs', 'Reddit Posts', 'Inst. %', 'Reddit %', 'Gap %']

        def color_gap(val):
            if val > 5:   return 'background-color: #d4edda; color: #155724'
            elif val < -5: return 'background-color: #f8d7da; color: #721c24'
            return ''

        st.dataframe(display_df.style.applymap(color_gap, subset=['Gap %']),
                     use_container_width=True, hide_index=True, height=300)

    st.divider()

    # ── SECTION 2: TREND CORRELATION ──────────────────────────────────────
    st.subheader("📈 Trend Correlation — Does Reddit Lead Institutional Sources?")
    trend_col1, trend_col2 = st.columns([1, 3])

    with trend_col1:
        sel_corr_category = st.selectbox("Select Category", options=sorted(combined['Category'].unique()),
                                          key="corr_category", format_func=lambda x: x.replace('_', ' '))
        corr_freq = st.selectbox("Frequency", ["Monthly", "Weekly"], key="corr_freq")

    with trend_col2:
        freq = {"Monthly": "ME", "Weekly": "W-SUN"}[corr_freq]
        inst_tags = [k for k, v in THEME_MAP.items() if v == sel_corr_category]

        inst_trend = (df[df['primary_tag_norm'] == sel_corr_category]
                      .groupby(pd.Grouper(key='date', freq=freq)).size().reset_index(name='Institutional'))

        reddit_trend = (reddit_posts[reddit_posts['theme'].isin(inst_tags)]
                        .groupby(pd.Grouper(key='run_date', freq=freq)).size().reset_index(name='Reddit'))
        reddit_trend.columns = ['date', 'Reddit']

        trend_merged = inst_trend.merge(reddit_trend, on='date', how='outer').fillna(0).sort_values('date')

        if not trend_merged.empty and trend_merged['Reddit'].sum() > 0:
            fig_corr = go.Figure()
            fig_corr.add_trace(go.Scatter(x=trend_merged['date'], y=trend_merged['Institutional'],
                                          name='Institutional', line=dict(color='#003366', width=2), mode='lines+markers'))
            fig_corr.add_trace(go.Scatter(x=trend_merged['date'], y=trend_merged['Reddit'],
                                          name='Reddit Posts', line=dict(color='#FF4500', width=2, dash='dot'),
                                          mode='lines+markers', yaxis='y2'))
            fig_corr.update_layout(template='plotly_white', height=260,
                                   legend=dict(orientation="h", y=1.1), margin=dict(t=10, b=10, l=0, r=60),
                                   yaxis=dict(title='Institutional Docs', side='left'),
                                   yaxis2=dict(title='Reddit Posts', side='right', overlaying='y'),
                                   hovermode='x unified')
            st.plotly_chart(fig_corr, use_container_width=True)

            if len(trend_merged) >= 4:
                inst_vals, reddit_vals = trend_merged['Institutional'].values, trend_merged['Reddit'].values
                if inst_vals.std() > 0 and reddit_vals.std() > 0:
                    corr = np.corrcoef(inst_vals, reddit_vals)[0, 1]
                    st.caption(f"📌 Pearson correlation: **{corr:.2f}**. "
                               f"{'Strong alignment — both sources tracking together.' if abs(corr) > 0.6 else 'Weak alignment — Reddit may be capturing signal not yet in formal sources.'}")
        else:
            st.info(f"Not enough Reddit data yet for {sel_corr_category.replace('_',' ')} to compute correlation. Run the pipeline daily to build volume.")

    st.divider()

    # ── SECTION 3: FRAUD CHATBOT ──────────────────────────────────────────
    st.subheader("💬 Fraud Intelligence Assistant")
    st.caption("Powered by phi3 running locally via Ollama — no API key required. Ask questions grounded in both institutional and Reddit data.")

    @st.cache_data
    def build_data_context(df_hash, reddit_hash):
        inst_summary   = df['primary_tag'].value_counts().head(10).to_dict()
        inst_sources   = df['source'].value_counts().to_dict()
        inst_date_range = f"{df['date'].min().date()} to {df['date'].max().date()}"
        total_inst     = len(df)
        reddit_theme_counts = reddit_posts['theme'].value_counts().to_dict()
        total_reddit   = len(reddit_posts)
        reddit_run_dates = sorted(reddit_posts['run_date'].dt.date.unique().astype(str).tolist())
        if not reddit_summaries.empty:
            latest_summaries = reddit_summaries.sort_values('run_date').groupby('theme').last()['theme_summary_text'].to_dict()
        else:
            latest_summaries = {}
        return f"""
INSTITUTIONAL DATA:
- Total documents: {total_inst:,} | Date range: {inst_date_range}
- Sources: {json.dumps(inst_sources)}
- Top fraud categories: {json.dumps(inst_summary)}

REDDIT DATA:
- Total posts: {total_reddit:,} | Pipeline runs: {reddit_run_dates}
- Posts per theme: {json.dumps(reddit_theme_counts)}

REDDIT THEME SUMMARIES:
{chr(10).join([f"- {t}: {s[:250]}..." for t, s in list(latest_summaries.items())[:8]])}

SIGNAL GAP (Reddit % minus Institutional %):
{combined.head(5)[['Category','Gap']].to_string(index=False)}
"""

    try:
        data_context = build_data_context(hash(str(df.shape)), hash(str(reddit_posts.shape)))
    except Exception:
        data_context = "Data context unavailable."

    # Suggested questions
    st.markdown("**Try asking:**")
    q_col1, q_col2, q_col3 = st.columns(3)
    suggested = [
        "Does Reddit data support the institutional trend in Identity Fraud?",
        "Which fraud themes show Reddit leading institutional sources?",
        "Summarize what Reddit says about crypto fraud vs formal sources.",
        "Which categories have the biggest gap between Reddit and institutional data?",
        "Are there fraud patterns on Reddit not yet in institutional sources?",
        "What fraud themes should analysts prioritize based on both data sources?",
    ]
    for i, q in enumerate(suggested[:3]):
        with [q_col1, q_col2, q_col3][i]:
            if st.button(q, key=f"sq_{i}", use_container_width=True):
                st.session_state.chat_input_prefill = q

    if "cross_chat_history" not in st.session_state:
        st.session_state.cross_chat_history = []

    for msg in st.session_state.cross_chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prefill    = st.session_state.pop("chat_input_prefill", "")
    user_input = st.chat_input("Ask a question about the fraud data...", key="cross_chat")

    if user_input or prefill:
        question = user_input or prefill

        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.cross_chat_history.append({"role": "user", "content": question})

        system_prompt = f"""You are a fraud intelligence analyst assistant.
You have access to two datasets:
1. INSTITUTIONAL DATA — FinCEN, FTC, FBI, IC3, BleepingComputer, Outseer, PYMNTS
2. REDDIT DATA — 24 fraud-relevant subreddits classified into 18 fraud themes

Rules:
- Only answer based on the data provided. Do not speculate beyond it.
- When Reddit and institutional sources agree, say so explicitly.
- When they diverge, flag it as an area for investigation.
- Be concise and actionable. Always cite which data source supports each claim.
- If data is insufficient to answer, say so clearly.

CURRENT DATA CONTEXT:
{data_context}"""

        history_text = ""
        for h in st.session_state.cross_chat_history[:-1]:
            role = "User" if h["role"] == "user" else "Assistant"
            history_text += f"{role}: {h['content']}\n"

        full_prompt = f"{system_prompt}\n\n{history_text}User: {question}\nAssistant:"

        with st.chat_message("assistant"):
            with st.spinner("Analyzing both data sources..."):
                try:
                    import requests
                    response = requests.post(
                        "http://localhost:11434/api/generate",
                        json={
                            "model": "phi3",
                            "prompt": full_prompt,
                            "stream": False,
                            "options": {"temperature": 0.3, "num_predict": 512}
                        },
                        timeout=120
                    )
                    if response.status_code == 200:
                        answer = response.json().get("response", "").strip() or "No response generated. Try rephrasing."
                    else:
                        answer = f"Ollama error {response.status_code}: {response.text[:200]}"

                    st.markdown(answer)
                    st.session_state.cross_chat_history.append({"role": "assistant", "content": answer})

                except requests.exceptions.ConnectionError:
                    err = (
                        "⚠️ **Ollama is not running.**\n\n"
                        "This tab requires a local Ollama server with the `phi3` model.\n\n"
                        "**To enable:**\n"
                        "1. Install Ollama: https://ollama.com/\n"
                        "2. Pull the model: `ollama pull phi3`\n"
                        "3. Start the server: `ollama serve`\n"
                        "4. Refresh this page\n\n"
                        "The other tabs work without Ollama."
                    )
                    st.error(err)
                    st.session_state.cross_chat_history.append({"role": "assistant", "content": err})
                except Exception as e:
                    err = f"Error: {str(e)}"
                    st.error(err)
                    st.session_state.cross_chat_history.append({"role": "assistant", "content": err})

    if st.session_state.cross_chat_history:
        if st.button("🗑️ Clear conversation", key="clear_cross_chat"):
            st.session_state.cross_chat_history = []
            st.rerun()

# ── FOOTER: TEAM ATTRIBUTION ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666; font-size: 0.85rem; padding: 1rem 0;">
        <strong>About this dashboard:</strong><br>
        This dashboard integrates a Reddit fraud intelligence pipeline (individual work) with multi-source institutional fraud data
        ingested by a graduate capstone team. The Reddit pipeline — including ingestion, classification, embedding, and LLM synthesis —
        is the individual contribution documented in this repository. The seven institutional-source pipelines (FinCEN, FTC, FBI, IC3,
        BleepingComputer, Outseer, PYMNTS) and the shared canonical schema were built by teammates as part of the 5-person capstone pod.
        See the README for full attribution.
    </div>
    """,
    unsafe_allow_html=True,
)
