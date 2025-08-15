import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import psycopg
from psycopg.rows import dict_row
import os
import json
from datetime import datetime

st.set_page_config(page_title="Document Classification Results", page_icon="📊", layout="wide")

RESULTS_DSN = os.environ.get("RESULTS_DSN", "")
RUN_ID = os.environ.get("RUN_ID", "")

if not RESULTS_DSN:
    st.error("Please set the RESULTS_DSN environment variable with your database connection string")
    st.code("export RESULTS_DSN='postgresql+psycopg://username:password@host:port/database'")
    st.code("streamlit run visuals.py")
    st.stop()

def pg_dsn(dsn: str) -> str:
    """Convert SQLAlchemy DSN to psycopg DSN"""
    return dsn.replace("postgresql+psycopg://", "postgresql://")

@st.cache_data(ttl=60)
def load_data(run_id=None):
    """Load classification data from database"""
    with psycopg.connect(pg_dsn(RESULTS_DSN), row_factory=dict_row) as conn:
        if run_id:
            run_query = """
                SELECT run_id, dag_id, ts_started, ts_finished, status, 
                       model_version, data_snapshot_uri, preds_uri
                FROM runs 
                WHERE run_id = %s
            """
            run_info = conn.execute(run_query, (run_id,)).fetchone()
        else:
            run_query = """
                SELECT run_id, dag_id, ts_started, ts_finished, status, 
                       model_version, data_snapshot_uri, preds_uri
                FROM runs 
                ORDER BY ts_started DESC 
                LIMIT 1
            """
            run_info = conn.execute(run_query).fetchone()
        
        if not run_info:
            return None, None, None
        
        class_query = """
            SELECT doc_id, source_file, predicted_type, dated, 
                   needs_human_review, confidence_reasoning,
                   model_version, created_at
            FROM classifications
            WHERE run_id = %s
        """
        classifications = conn.execute(class_query, (run_info['run_id'],)).fetchall()

        stats_query = """
            SELECT predicted_type, 
                   COUNT(*) as count,
                   SUM(CASE WHEN needs_human_review THEN 1 ELSE 0 END) as review_count
            FROM classifications
            WHERE run_id = %s
            GROUP BY predicted_type
            ORDER BY count DESC
        """
        type_stats = conn.execute(stats_query, (run_info['run_id'],)).fetchall()
        
        return run_info, classifications, type_stats

st.title("📊 Document Classification Results")

if not RESULTS_DSN:
    st.stop()

try:
    run_info, classifications, type_stats = load_data(RUN_ID)
except Exception as e:
    st.error(f"Database connection failed: {e}")
    st.info("Please check your database connection string and ensure the database is running.")
    st.stop()

if not run_info:
    st.error("No run data found. Please check the database connection and run ID.")
    st.stop()

st.header("Run Information")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Run ID", run_info['run_id'][:16] + "...")
with col2:
    st.metric("Status", run_info['status'].upper())
with col3:
    st.metric("Model", run_info['model_version'])
with col4:
    if run_info['ts_finished']:
        duration = (run_info['ts_finished'] - run_info['ts_started']).total_seconds()
        st.metric("Duration", f"{duration:.1f}s")
    else:
        st.metric("Duration", "Running...")

if classifications:
    df = pd.DataFrame(classifications)
    
    st.header("Summary Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Documents", len(df))
    with col2:
        st.metric("Document Types", df['predicted_type'].nunique())
    with col3:
        review_count = df['needs_human_review'].sum()
        st.metric("Need Review", f"{review_count} ({review_count/len(df)*100:.1f}%)")
    with col4:
        st.metric("Auto-approved", len(df) - review_count)

    st.header("Visualizations")
    
    types = [t['predicted_type'] for t in type_stats]
    counts = [t['count'] for t in type_stats]
    review_counts = [t['review_count'] for t in type_stats]
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        colors = plt.cm.Set3(range(len(types)))
        wedges, texts, autotexts = ax1.pie(counts, labels=types, autopct='%1.1f%%', 
                                            colors=colors, startangle=90)
        ax1.set_title('Document Type Distribution')

        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        st.pyplot(fig1)
    
    with col2:

        fig2, ax2 = plt.subplots(figsize=(8, 6))
        bars = ax2.bar(range(len(types)), counts, color='skyblue')
        ax2.set_xticks(range(len(types)))
        ax2.set_xticklabels(types, rotation=45, ha='right')
        ax2.set_xlabel('Document Type')
        ax2.set_ylabel('Count')
        ax2.set_title('Documents by Type')
        ax2.grid(axis='y', alpha=0.3)
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        st.pyplot(fig2)
    

    st.subheader("Review Status Analysis")
    
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    
    x = range(len(types))
    width = 0.35
    
    auto_approved = [counts[i] - review_counts[i] for i in range(len(counts))]
    
    bars1 = ax3.bar([i - width/2 for i in x], auto_approved, width, 
                    label='Auto-approved', color='#2ecc71', alpha=0.8)
    bars2 = ax3.bar([i + width/2 for i in x], review_counts, width, 
                    label='Need Review', color='#e74c3c', alpha=0.8)
    
    ax3.set_xlabel('Document Type')
    ax3.set_ylabel('Number of Documents')
    ax3.set_title('Review Status by Document Type')
    ax3.set_xticks(x)
    ax3.set_xticklabels(types, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    st.pyplot(fig3)
    
    st.header("Detailed Results")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        type_filter = st.multiselect("Filter by Type", options=df['predicted_type'].unique())
    with col2:
        review_filter = st.selectbox("Review Status", ["All", "Need Review", "Auto-approved"])
    with col3:
        search = st.text_input("Search in source file")
    
    # Apply filters
    filtered_df = df.copy()
    if type_filter:
        filtered_df = filtered_df[filtered_df['predicted_type'].isin(type_filter)]
    if review_filter == "Need Review":
        filtered_df = filtered_df[filtered_df['needs_human_review'] == True]
    elif review_filter == "Auto-approved":
        filtered_df = filtered_df[filtered_df['needs_human_review'] == False]
    if search:
        filtered_df = filtered_df[filtered_df['source_file'].str.contains(search, case=False, na=False)]
    
    display_df = filtered_df[['doc_id', 'source_file', 'predicted_type', 'needs_human_review', 'confidence_reasoning']].copy()
    
    display_df['confidence_reasoning'] = display_df['confidence_reasoning'].apply(
        lambda x: x[:200] + '...' if x and len(x) > 200 else x
    )
    
    st.dataframe(
        display_df,
        use_container_width=True,
        column_config={
            "doc_id": "Document ID",
            "source_file": "Source File",
            "predicted_type": "Type",
            "needs_human_review": "Needs Review",
            "confidence_reasoning": st.column_config.TextColumn(
                "Confidence Reasoning",
                help="Model's reasoning for the classification",
                width="large"
            )
        }
    )
    
    st.caption(f"Showing {len(filtered_df)} of {len(df)} documents")
    
    if len(filtered_df) > 0:
        with st.expander("View Full Confidence Reasoning"):
            selected_doc = st.selectbox(
                "Select a document to view full reasoning:",
                options=filtered_df['doc_id'].tolist(),
                format_func=lambda x: f"{x} - {filtered_df[filtered_df['doc_id']==x]['source_file'].values[0]}"
            )
            if selected_doc:
                full_reasoning = filtered_df[filtered_df['doc_id'] == selected_doc]['confidence_reasoning'].values[0]
                st.text_area("Full Reasoning", full_reasoning, height=200, disabled=True)
    
else:
    st.warning("No classification data found for this run.")

st.markdown("---")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")