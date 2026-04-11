"""
Streamlit Web Application for Learning Agent System

A professional, modern web interface for the autonomous learning agent.
Provides interactive learning experiences with checkpoint progression,
dynamic materials, and adaptive teaching.
"""

import streamlit as st
import asyncio
import sys
from pathlib import Path
from typing import List, Dict, Optional
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.sample_data import create_learning_paths
from src.custom_topics import load_custom_topics, create_topic_wizard, add_custom_topic
from src.multi_checkpoint import run_multi_checkpoint_session
from src.models import LearningPath, Checkpoint
from src.workflow import create_unified_workflow, create_question_generation_workflow
from src.models import LearningAgentState
from src.user_interaction import collect_user_answers, display_score_feedback
from src.file_upload import get_upload_handler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Learning Agent System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Design System ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}
.main .block-container {
    padding: 1.5rem 2.5rem 3rem !important;
    max-width: 1280px !important;
}

/* ── App Header ── */
.app-header {
    background: linear-gradient(135deg, #064e3b 0%, #065f46 40%, #047857 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 0.5rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 24px rgba(6,78,59,0.28);
}
.app-header::after {
    content: '';
    position: absolute;
    top: -40%; right: -5%;
    width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(255,255,255,0.07) 0%, transparent 65%);
    border-radius: 50%;
    pointer-events: none;
}
.app-header-title {
    font-size: 2rem; font-weight: 800; color: #ffffff;
    margin: 0 0 0.2rem; line-height: 1.15;
}
.app-header-sub { color: #a7f3d0; font-size: 0.9rem; margin: 0 0 0.85rem; }
.header-tag {
    display: inline-flex; align-items: center; gap: 5px;
    background: rgba(255,255,255,0.13); border: 1px solid rgba(255,255,255,0.2);
    padding: 0.25rem 0.75rem; border-radius: 100px;
    color: #d1fae5; font-size: 0.75rem; font-weight: 500; margin-right: 0.4rem;
}

/* ── Status Pills ── */
.status-online  { display:inline-flex;align-items:center;gap:5px;padding:0.3rem 0.8rem;border-radius:100px;background:#dcfce7;color:#166534;border:1px solid #bbf7d0;font-size:0.8rem;font-weight:500; }
.status-warning { display:inline-flex;align-items:center;gap:5px;padding:0.3rem 0.8rem;border-radius:100px;background:#fef3c7;color:#92400e;border:1px solid #fde68a;font-size:0.8rem;font-weight:500; }
.status-offline { display:inline-flex;align-items:center;gap:5px;padding:0.3rem 0.8rem;border-radius:100px;background:#fee2e2;color:#991b1b;border:1px solid #fecaca;font-size:0.8rem;font-weight:500; }

/* ── Breadcrumb ── */
.breadcrumb {
    display: flex; align-items: center; gap: 6px;
    padding: 0.55rem 1rem; background: #f0fdf4;
    border: 1px solid #bbf7d0; border-radius: 8px;
    margin: 0.75rem 0 1.5rem; font-size: 0.82rem; flex-wrap: wrap;
}
.bc-step { color: #94a3b8; display: flex; align-items: center; gap: 5px; }
.bc-step.done   { color: #10b981; }
.bc-step.active { color: #059669; font-weight: 600; }
.bc-sep { color: #cbd5e1; font-size: 0.7rem; }

/* ── Path Cards ── */
.path-card {
    background: #fff; border: 1px solid #e2e8f0; border-radius: 12px;
    overflow: hidden; margin-bottom: 0.6rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    transition: box-shadow 0.2s, border-color 0.2s, transform 0.15s;
}
.path-card:hover {
    border-color: #059669;
    box-shadow: 0 6px 20px rgba(5,150,105,0.14);
    transform: translateY(-2px);
}
.pc-bar     { height: 5px; background: linear-gradient(90deg, #059669, #10b981); }
.pc-bar.c1  { background: linear-gradient(90deg, #6366f1, #8b5cf6); }
.pc-bar.c2  { background: linear-gradient(90deg, #f59e0b, #ef4444); }
.pc-bar.c3  { background: linear-gradient(90deg, #06b6d4, #3b82f6); }
.pc-bar.c4  { background: linear-gradient(90deg, #ec4899, #f43f5e); }
.pc-body    { padding: 1.2rem 1.4rem 0.9rem; }
.pc-title   { font-size: 1.05rem; font-weight: 700; color: #0f172a; margin: 0 0 0.4rem; }
.pc-desc    { font-size: 0.85rem; color: #64748b; line-height: 1.55; margin: 0 0 0.75rem; }
.pc-meta    { display: flex; align-items: center; gap: 8px; }
.tag        { display:inline-flex;align-items:center;gap:3px;background:#f1f5f9;color:#475569;padding:0.15rem 0.55rem;border-radius:6px;font-size:0.72rem;font-weight:600; }
.tag.custom { background: #fdf4ff; color: #7c3aed; }

/* ── Action Box ── */
.action-box {
    background: linear-gradient(135deg,#f0fdf4,#ecfdf5);
    border: 1px solid #a7f3d0; border-radius: 12px; padding: 1.5rem;
}
.action-box h3 { font-size: 1rem; font-weight: 700; color: #064e3b; margin: 0 0 0.4rem; }
.action-box p  { font-size: 0.82rem; color: #065f46; margin: 0 0 1rem; line-height: 1.5; }

/* ── Form Card ── */
.form-card {
    background: #fff; border: 1px solid #e2e8f0; border-radius: 12px;
    padding: 2rem 2.5rem; margin-top: 1.5rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06);
}
.form-card h3 { font-size: 1.1rem; font-weight: 700; color: #0f172a; margin: 0 0 0.4rem; }
.form-card p  { font-size: 0.85rem; color: #64748b; margin: 0 0 1.5rem; }

/* ── Upload Zone ── */
.upload-zone {
    background: #f8fafc; border: 2px dashed #cbd5e1; border-radius: 12px;
    padding: 1.75rem; text-align: center; margin-bottom: 1.25rem;
}
.upload-zone p { color: #64748b; font-size: 0.875rem; margin: 0.35rem 0 0; }
.ft-pill { display:inline-flex;align-items:center;background:#e2e8f0;color:#475569;padding:0.2rem 0.55rem;border-radius:5px;font-size:0.72rem;font-weight:700;margin:0.2rem; }
.file-item { display:flex;align-items:center;gap:8px;padding:0.55rem 1rem;background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;margin-bottom:0.4rem;font-size:0.85rem;color:#166534; }

/* ── Study Reader ── */
.reader-card {
    background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px;
    padding: 2.5rem 3rem; box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    font-size: 16px; line-height: 1.9; color: #1e293b;
}
.reader-stat { display:inline-flex;align-items:center;gap:5px;background:#f1f5f9;color:#475569;padding:0.25rem 0.7rem;border-radius:6px;font-size:0.78rem;font-weight:500;margin-right:0.5rem; }
.res-card { background:#f8fafc;border:1px solid #e2e8f0;border-left:3px solid #3b82f6;border-radius:8px;padding:12px 16px;margin-bottom:8px;font-size:0.875rem; }
.res-num  { display:inline-flex;align-items:center;justify-content:center;width:22px;height:22px;background:#3b82f6;color:#fff;border-radius:50%;font-size:0.7rem;font-weight:700;margin-right:8px; }

/* ── Checkpoint Header ── */
.cp-header {
    background: linear-gradient(135deg,#f0fdf4,#ecfdf5);
    border: 1px solid #a7f3d0; border-radius: 12px;
    padding: 1.25rem 1.5rem; margin-bottom: 1.25rem;
}
.cp-label { font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:#059669;margin:0 0 0.25rem; }
.cp-title { font-size:1.3rem;font-weight:700;color:#0f172a;margin:0 0 0.3rem; }
.cp-desc  { font-size:0.875rem;color:#64748b;margin:0; }
.progress-dots { display:flex;gap:5px;margin-top:0.85rem;flex-wrap:wrap; }
.pd        { width:28px;height:7px;border-radius:3px;background:#e2e8f0; }
.pd.done   { background:#10b981; }
.pd.active { background:#059669; }

/* ── Question Cards ── */
.q-card { background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:1.5rem;margin-bottom:1.25rem;box-shadow:0 1px 5px rgba(0,0,0,0.05); }
.q-header { display:flex;align-items:center;gap:8px;margin-bottom:0.85rem; }
.q-num    { display:inline-flex;align-items:center;justify-content:center;width:28px;height:28px;background:#059669;color:#fff;border-radius:50%;font-size:0.8rem;font-weight:700;flex-shrink:0; }
.q-type-mcq  { display:inline-flex;align-items:center;padding:0.15rem 0.55rem;border-radius:5px;background:#ede9fe;color:#5b21b6;font-size:0.7rem;font-weight:700;text-transform:uppercase;letter-spacing:.05em; }
.q-type-open { display:inline-flex;align-items:center;padding:0.15rem 0.55rem;border-radius:5px;background:#dbeafe;color:#1e40af;font-size:0.7rem;font-weight:700;text-transform:uppercase;letter-spacing:.05em; }
.q-text { font-size:1rem;font-weight:500;color:#1e293b;line-height:1.65;padding:0.85rem 1.1rem;background:#f8fafc;border-radius:8px;border-left:3px solid #059669;margin-bottom:1rem; }

/* ── Results ── */
.score-hero { text-align:center;padding:2.5rem 2rem;background:#fff;border:1px solid #e2e8f0;border-radius:16px;box-shadow:0 4px 20px rgba(0,0,0,0.07); }
.score-num  { font-size:5rem;font-weight:800;line-height:1; }
.score-num.pass { color:#059669; }
.score-num.warn { color:#d97706; }
.score-num.fail { color:#dc2626; }
.score-sub  { font-size:0.8rem;font-weight:500;color:#94a3b8;text-transform:uppercase;letter-spacing:.07em;margin-top:0.2rem; }
.verdict    { display:inline-flex;align-items:center;gap:6px;padding:0.5rem 1.5rem;border-radius:100px;font-size:0.9rem;font-weight:600;margin-top:1rem; }
.verdict.pass { background:#dcfce7;color:#166534;border:1px solid #bbf7d0; }
.verdict.fail { background:#fef3c7;color:#92400e;border:1px solid #fde68a; }
.r-score    { display:inline-flex;align-items:center;padding:0.18rem 0.6rem;border-radius:100px;font-size:0.78rem;font-weight:600;float:right; }
.rs-pass { background:#dcfce7;color:#166534; }
.rs-mid  { background:#fef3c7;color:#92400e; }
.rs-fail { background:#fee2e2;color:#991b1b; }

/* ── Completion ── */
.completion-hero {
    text-align:center;padding:3.5rem 2rem;
    background:linear-gradient(135deg,#064e3b,#047857);
    border-radius:20px;color:#fff;
    box-shadow:0 8px 30px rgba(6,78,59,0.3);
    margin-bottom:1.5rem;position:relative;overflow:hidden;
}
.completion-hero::after { content:'';position:absolute;top:-30%;right:-5%;width:300px;height:300px;background:radial-gradient(circle,rgba(255,255,255,0.06) 0%,transparent 65%);border-radius:50%;pointer-events:none; }
.completion-hero h2 { font-size:2.2rem;font-weight:800;color:#fff;margin:0 0 0.4rem; }
.completion-hero p  { color:#a7f3d0;font-size:1rem;margin:0; }
.stat-box { background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:1.5rem;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,0.06); }
.stat-v   { font-size:2.75rem;font-weight:800;color:#059669;line-height:1;margin-bottom:0.3rem; }
.stat-l   { font-size:0.75rem;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:.06em; }

/* ── Notice Boxes ── */
.success-box { padding:1rem 1.25rem;background:#f0fdf4;border:1px solid #bbf7d0;border-left:4px solid #22c55e;border-radius:8px;color:#166534;font-size:0.875rem;margin-bottom:0.75rem; }
.warning-box { padding:1rem 1.25rem;background:#fffbeb;border:1px solid #fde68a;border-left:4px solid #f59e0b;border-radius:8px;color:#92400e;font-size:0.875rem;margin-bottom:0.75rem; }
.error-box   { padding:1rem 1.25rem;background:#fef2f2;border:1px solid #fecaca;border-left:4px solid #ef4444;border-radius:8px;color:#991b1b;font-size:0.875rem;margin-bottom:0.75rem; }
.info-box    { padding:1rem 1.25rem;background:#eff6ff;border:1px solid #bfdbfe;border-left:4px solid #3b82f6;border-radius:8px;color:#1e40af;font-size:0.875rem;margin-bottom:0.75rem; }

/* ── Global Overrides ── */
.stProgress > div > div > div > div { background:linear-gradient(90deg,#059669,#10b981) !important;border-radius:4px !important; }
.stButton > button { border-radius:8px !important; font-weight:500 !important; }
[data-testid="metric-container"] { background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:1rem 1.25rem !important;box-shadow:0 1px 4px rgba(0,0,0,0.05); }
#MainMenu { visibility:hidden; }
footer { visibility:hidden; }
header[data-testid="stHeader"] { background:transparent; }
</style>
""", unsafe_allow_html=True)

# ── UI Helpers ────────────────────────────────────────────────────────────────
STAGE_ORDER  = ['select_path', 'upload_files', 'study', 'learning', 'results']
STAGE_LABELS = {
    'select_path':  '🗺️ Choose Path',
    'upload_files': '📁 Materials',
    'study':        '📖 Study',
    'learning':     '🧠 Assessment',
    'results':      '📊 Results',
}
CARD_COLORS = ['', 'c1', 'c2', 'c3', 'c4']

def get_path_icon(title: str) -> str:
    t = title.lower()
    if any(k in t for k in ['machine learning', ' ml ']):  return '🤖'
    if any(k in t for k in ['deep learning', 'neural']):   return '🧬'
    if any(k in t for k in ['data science', 'analytics']): return '📊'
    if any(k in t for k in ['nlp', 'natural language']):   return '💬'
    if 'python' in t:                                       return '🐍'
    if any(k in t for k in ['web', 'react', 'javascript']): return '🌐'
    if any(k in t for k in ['math', 'statistics', 'algebra']): return '📐'
    return '🎯'

# Initialize session state
def init_session_state():
    """Initialize session state variables."""
    if 'stage' not in st.session_state:
        st.session_state.stage = 'select_path'  # select_path, upload_files, learning, results
    
    if 'selected_path' not in st.session_state:
        st.session_state.selected_path = None
    
    if 'current_checkpoint_index' not in st.session_state:
        st.session_state.current_checkpoint_index = 0
    
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    
    if 'learning_state' not in st.session_state:
        st.session_state.learning_state = None
    
    if 'questions' not in st.session_state:
        st.session_state.questions = []
    
    if 'answers' not in st.session_state:
        st.session_state.answers = {}
    
    if 'show_create_topic' not in st.session_state:
        st.session_state.show_create_topic = False
    
    if 'completed_checkpoints' not in st.session_state:
        st.session_state.completed_checkpoints = []

    if 'study_materials_ready' not in st.session_state:
        st.session_state.study_materials_ready = False

def render_header():
    """Render application header with gradient banner, Ollama status, and breadcrumb."""
    # Gradient hero banner
    st.markdown("""
    <div class="app-header">
        <div class="app-header-title">📚 Learning Agent System</div>
        <div class="app-header-sub">Autonomous AI-powered adaptive learning with Feynman Technique &amp; LangGraph</div>
        <div>
            <span class="header-tag">⚡ LangGraph</span>
            <span class="header-tag">🤖 llama3.1</span>
            <span class="header-tag">🧠 ChromaDB</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Status pill + breadcrumb row
    col_st, col_bc = st.columns([1, 4])
    with col_st:
        try:
            import httpx
            response = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
            if response.status_code == 200:
                st.markdown('<span class="status-online">🟢 Ollama Online</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="status-warning">🟡 Connection Issue</span>', unsafe_allow_html=True)
        except:
            st.markdown('<span class="status-offline">🔴 Ollama Offline</span>', unsafe_allow_html=True)
            with st.expander("⚠️ How to start Ollama"):
                st.code("ollama serve", language="bash")
                st.markdown("Ensure Ollama is running in the background.")

    with col_bc:
        stage = st.session_state.get('stage', 'select_path')
        current_idx = STAGE_ORDER.index(stage) if stage in STAGE_ORDER else 0
        parts = []
        for i, s in enumerate(STAGE_ORDER):
            cls = 'done' if i < current_idx else ('active' if i == current_idx else '')
            if i > 0:
                parts.append('<span class="bc-sep">›</span>')
            parts.append(f'<span class="bc-step {cls}">{STAGE_LABELS[s]}</span>')
        st.markdown(f'<div class="breadcrumb">{"".join(parts)}</div>', unsafe_allow_html=True)

def render_path_selection():
    """Render learning path selection interface."""
    st.markdown("""
    <div style="margin-bottom:1.25rem;">
        <div style="font-size:1.5rem;font-weight:700;color:#0f172a;margin:0 0 0.3rem;">Choose Your Learning Path</div>
        <div style="color:#64748b;font-size:0.875rem;">Select a structured curriculum or create your own custom topic</div>
    </div>
    """, unsafe_allow_html=True)

    default_paths = create_learning_paths()
    custom_paths  = load_custom_topics()
    all_paths     = default_paths + custom_paths

    col1, col2 = st.columns([3, 1])

    with col1:
        for i, path in enumerate(all_paths):
            is_custom  = i >= len(default_paths)
            color_cls  = CARD_COLORS[i % len(CARD_COLORS)]
            icon       = get_path_icon(path['title'])
            n_checks   = len(path['checkpoints'])
            custom_tag = '<span class="tag custom">✨ Custom</span>' if is_custom else ''

            st.markdown(f"""
            <div class="path-card">
                <div class="pc-bar {color_cls}"></div>
                <div class="pc-body">
                    <div class="pc-title">{icon} {path['title']}</div>
                    <div class="pc-desc">{path['description']}</div>
                    <div class="pc-meta">
                        <span class="tag">📋 {n_checks} checkpoint{'s' if n_checks != 1 else ''}</span>
                        {custom_tag}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if st.button(f"Start Learning →", key=f"select_{i}", use_container_width=True):
                st.session_state.selected_path = path
                st.session_state.stage = 'upload_files'
                st.rerun()

    with col2:
        st.markdown("""
        <div class="action-box">
            <h3>✏️ Custom Topic</h3>
            <p>Don't see your topic? Create a custom learning path for any subject you want to master.</p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("➕ Create Custom Topic", use_container_width=True, type="primary"):
            st.session_state.show_create_topic = True
            st.rerun()

        st.markdown(f"""
        <div style="margin-top:1rem;padding:1rem;background:#f8fafc;border-radius:10px;border:1px solid #e2e8f0;text-align:center;">
            <div style="font-size:2rem;font-weight:800;color:#059669;line-height:1;">{len(all_paths)}</div>
            <div style="font-size:0.72rem;color:#64748b;font-weight:600;text-transform:uppercase;letter-spacing:.05em;margin-top:0.2rem;">Paths Available</div>
        </div>
        """, unsafe_allow_html=True)

    if st.session_state.show_create_topic:
        render_custom_topic_form()

def render_custom_topic_form():
    """Render custom topic creation form."""
    st.markdown("<hr style='border:none;border-top:1px solid #e2e8f0;margin:1.5rem 0;'>", unsafe_allow_html=True)

    st.markdown("""
    <div class="form-card">
        <h3>✏️ Create Custom Learning Topic</h3>
        <p>Enter any topic and the AI will automatically generate study materials, assessment questions, and adaptive explanations.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="info-box">💡 <strong>Tip:</strong> Be specific — e.g., "Python Decorators", "Gradient Descent", "REST API Design".</div>', unsafe_allow_html=True)

    with st.form("custom_topic_form"):
        topic_name = st.text_input(
            "Topic Name",
            placeholder="e.g., Python Basics, Machine Learning, Web Development",
            help="Enter any topic you want to learn about"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            submitted = st.form_submit_button("🚀 Start Learning", use_container_width=True, type="primary")
        with col2:
            cancelled = st.form_submit_button("✕ Cancel", use_container_width=True)
        
        if submitted and topic_name and topic_name.strip():
            # Auto-generate topic ID from name
            topic_id = topic_name.lower().replace(" ", "_").replace("-", "_")
            # Remove special characters
            topic_id = ''.join(c for c in topic_id if c.isalnum() or c == '_')
            
            # Auto-generate description
            description = f"Learn the fundamentals and key concepts of {topic_name}"
            
            # Create a single checkpoint for the custom topic
            # The system will dynamically generate materials
            new_topic = {
                "id": topic_id,
                "title": topic_name.strip(),
                "description": description,
                "checkpoints": [
                    {
                        "id": f"{topic_id}_main",
                        "title": topic_name.strip(),
                        "description": description,
                        "requirements": [
                            f"Understand the fundamentals of {topic_name}",
                            f"Apply key concepts from {topic_name}",
                            f"Demonstrate knowledge of {topic_name}"
                        ]
                    }
                ]
            }
            
            if add_custom_topic(new_topic):
                st.success(f"Topic '{topic_name}' created! Starting learning session...")
                st.session_state.show_create_topic = False
                st.session_state.selected_path = new_topic
                st.session_state.stage = 'upload_files'
                st.rerun()
            else:
                st.error("This topic already exists. Please choose a different name.")
        
        if cancelled:
            st.session_state.show_create_topic = False
            st.rerun()

def render_file_upload():
    """Render file upload interface."""
    path = st.session_state.selected_path
    icon = get_path_icon(path['title'])
    n_checks = len(path['checkpoints'])

    # Selected-path context banner
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:12px;padding:0.75rem 1.25rem;
                background:#f0fdf4;border:1px solid #a7f3d0;border-radius:10px;margin-bottom:1.25rem;">
        <span style="font-size:1.6rem;">{icon}</span>
        <div style="flex:1;">
            <div style="font-size:0.7rem;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:#059669;">Selected Path</div>
            <div style="font-size:1rem;font-weight:700;color:#0f172a;">{path['title']}</div>
        </div>
        <span style="background:#059669;color:#fff;padding:0.2rem 0.65rem;border-radius:6px;font-size:0.72rem;font-weight:700;">
            {n_checks} checkpoint{'s' if n_checks != 1 else ''}
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:1.3rem;font-weight:700;color:#0f172a;margin-bottom:0.3rem;">📁 Upload Learning Materials</div>
    <div style="font-size:0.875rem;color:#64748b;margin-bottom:1.25rem;">
        Optionally upload your own documents — or skip to let AI fetch and generate materials automatically.
    </div>
    """, unsafe_allow_html=True)

    # Upload zone visual (decorative — actual uploader below)
    st.markdown("""
    <div class="upload-zone">
        <div style="font-size:2.5rem;">📄</div>
        <p>Drag &amp; drop files or use the uploader below</p>
        <div style="margin-top:0.75rem;">
            <span class="ft-pill">PDF</span>
            <span class="ft-pill">DOCX</span>
            <span class="ft-pill">MD</span>
            <span class="ft-pill">TXT</span>
        </div>
        <p style="font-size:0.72rem;color:#94a3b8;margin-top:0.5rem;">Maximum 10 MB per file</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload files",
        type=['pdf', 'docx', 'md', 'txt'],
        accept_multiple_files=True,
        help="Maximum 10MB per file",
        label_visibility="collapsed"
    )

    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        st.markdown(f'<div class="success-box">✅ <strong>{len(uploaded_files)} file(s)</strong> ready to use</div>', unsafe_allow_html=True)
        for file in uploaded_files:
            st.markdown(
                f'<div class="file-item">📄 <strong>{file.name}</strong>'
                f'<span style="margin-left:auto;color:#94a3b8;font-size:0.8rem;">{file.size / 1024:.1f} KB</span></div>',
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        if st.button("📂 Continue with Uploaded Files", use_container_width=True,
                     disabled=not uploaded_files, type="primary"):
            st.session_state.stage = 'learning'
            st.rerun()

    with col2:
        if st.button("⚡ Skip — Generate with AI", use_container_width=True):
            st.session_state.uploaded_files = []
            st.session_state.stage = 'learning'
            st.rerun()

def render_study_materials():
    """Display learning materials for study before assessment."""
    state = st.session_state.learning_state

    if not state:
        st.error("No learning materials available.")
        return

    summary   = state.get('summary', '')
    materials = state.get('collected_materials', [])

    st.markdown("""
    <div style="font-size:1.3rem;font-weight:700;color:#0f172a;margin-bottom:0.3rem;">📖 Study Materials</div>
    <div style="font-size:0.875rem;color:#64748b;margin-bottom:1.25rem;">
        Read through the material carefully — your assessment follows immediately after.
    </div>
    """, unsafe_allow_html=True)

    if summary:
        word_count = len(summary.split())
        read_time  = max(1, word_count // 200)

        st.markdown(f"""
        <div style="margin-bottom:1rem;">
            <span class="reader-stat">📝 {word_count} words</span>
            <span class="reader-stat">⏱️ ~{read_time} min read</span>
        </div>
        """, unsafe_allow_html=True)

        formatted = summary.replace('\n\n', '</p><p style="margin-top:16px;">').replace('\n', '<br>')
        st.markdown(f"""
        <div class="reader-card">
            <p style="margin-top:0;">{formatted}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="warning-box">⚠️ No learning content available.</div>', unsafe_allow_html=True)

    # Resource links
    resource_links = []
    for material in materials:
        url    = material.get('url') or material.get('link') or material.get('source_url')
        title  = material.get('title', 'Resource')
        source = material.get('source', 'Unknown')
        if url and url.startswith('http'):
            resource_links.append({'title': title, 'url': url, 'source': source})

    if resource_links:
        st.markdown(f"""
        <div style="font-size:1.1rem;font-weight:700;color:#0f172a;margin:1.75rem 0 0.75rem;">
            🔗 Additional Resources
            <span style="font-size:0.8rem;font-weight:500;color:#64748b;margin-left:6px;">({len(resource_links)} links)</span>
        </div>
        """, unsafe_allow_html=True)

        for i, resource in enumerate(resource_links, 1):
            st.markdown(f"""
            <div class="res-card">
                <span class="res-num">{i}</span>
                <strong style="color:#0f172a;">{resource['title']}</strong><br>
                <a href="{resource['url']}" target="_blank"
                   style="color:#2563eb;font-size:0.82rem;text-decoration:none;">🔗 {resource['url']}</a>
                <div style="font-size:0.72rem;color:#94a3b8;margin-top:4px;">Source: {resource['source']}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("✅ I'm Ready — Start Assessment", use_container_width=True, type="primary"):
            st.session_state.stage = 'learning'
            st.rerun()

def render_learning_session():
    """Render the learning session with materials, questions, and assessment."""
    path           = st.session_state.selected_path
    checkpoint_idx = st.session_state.current_checkpoint_index
    total          = len(path['checkpoints'])

    if checkpoint_idx >= total:
        render_completion()
        return

    checkpoint = path['checkpoints'][checkpoint_idx]

    # Checkpoint header card with animated progress dots
    dots = "".join(
        f'<div class="pd {"done" if d < checkpoint_idx else ("active" if d == checkpoint_idx else "")}"></div>'
        for d in range(total)
    )
    st.markdown(f"""
    <div class="cp-header">
        <div class="cp-label">Checkpoint {checkpoint_idx + 1} of {total}</div>
        <div class="cp-title">{checkpoint['title']}</div>
        <div class="cp-desc">{checkpoint['description']}</div>
        <div class="progress-dots">{dots}</div>
    </div>
    """, unsafe_allow_html=True)

    # Native progress bar (accessibility + percentage label)
    st.progress(checkpoint_idx / total,
                text=f"Progress: {checkpoint_idx}/{total} checkpoints completed")

    # Learning objectives
    with st.expander("📋 Learning Objectives", expanded=False):
        for req in checkpoint['requirements']:
            st.markdown(f"- {req}")
    
    # Run workflow in background
    if st.session_state.learning_state is None:
        with st.spinner("Preparing learning materials..."):
            # Process uploaded files
            uploaded_paths = None
            if st.session_state.uploaded_files:
                upload_handler = get_upload_handler()
                temp_paths = []
                
                for uploaded_file in st.session_state.uploaded_files:
                    file_content = uploaded_file.read()
                    result = upload_handler.handle_upload(
                        file_content=file_content,
                        file_name=uploaded_file.name
                    )
                    if result['success']:
                        temp_paths.append(result['file_path'])
                
                uploaded_paths = temp_paths if temp_paths else None
            
            # Initialize state
            initial_state: LearningAgentState = {
                "learning_path": path,
                "current_checkpoint_index": checkpoint_idx,
                "completed_checkpoints": st.session_state.completed_checkpoints,
                "total_checkpoints": len(path['checkpoints']),
                "has_next_checkpoint": checkpoint_idx < len(path['checkpoints']) - 1,
                "learning_path_completed": False,
                "current_checkpoint": checkpoint,
                "collected_materials": [],
                "summary": "",
                "milestone1_score": 0.0,
                "processed_context": [],
                "generated_questions": [],
                "verification_results": [],
                "score_percentage": 0.0,
                "meets_threshold": False,
                "feynman_retry_count": 0,
                "feynman_retry_requested": False,
                "feynman_explanations": [],
                "user_uploaded_notes_path": uploaded_paths,
                "materials_validation": None,
                "workflow_step": "initialized",
                "workflow_history": [],
                "errors": [],
                "user_mode": "streamlit"  # Flag to skip CLI input collection
            }
            
            # Run workflow up to question generation only (Streamlit mode)
            # This stops before verify_understanding so UI can collect answers
            workflow = create_question_generation_workflow()
            
            try:
                with st.status("Processing learning session...", expanded=True) as status:
                    st.write("Collecting materials...")
                    result = asyncio.run(workflow.ainvoke(initial_state))
                    
                    st.session_state.learning_state = result
                    st.session_state.questions = result.get('generated_questions', [])
                    
                    if st.session_state.questions:
                        st.write(f"Generated {len(st.session_state.questions)} questions")
                        st.session_state.study_materials_ready = True
                        status.update(label="Study materials ready!", state="complete")
                        # Redirect to study stage to show materials before assessment
                        st.session_state.stage = 'study'
                    else:
                        st.write("No questions generated")
                        status.update(label="Error", state="error")
                        
            except Exception as e:
                st.error(f"Error running workflow: {e}")
                logger.exception("Workflow error")
                return
            
            # After processing, redirect to study stage
            if st.session_state.study_materials_ready:
                st.rerun()
    
    # Display questions only if user has reviewed materials (coming from study stage)
    if st.session_state.questions and not st.session_state.answers:
        st.markdown("---")
        render_questions()
    elif st.session_state.answers:
        st.info("Answers submitted. Evaluating...")
    else:
        st.warning("No questions generated. Please try again or contact support.")

def render_questions():
    """Render interactive questions."""
    questions = st.session_state.questions

    st.markdown(f"""
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:1.25rem;">
        <div>
            <div style="font-size:1.25rem;font-weight:700;color:#0f172a;">🧠 Assessment</div>
            <div style="font-size:0.85rem;color:#64748b;">{len(questions)} questions · answer all to submit</div>
        </div>
        <span style="background:#f0fdf4;border:1px solid #a7f3d0;color:#059669;
                     padding:0.35rem 0.9rem;border-radius:8px;font-size:0.82rem;font-weight:600;">
            Passing threshold: 70%
        </span>
    </div>
    """, unsafe_allow_html=True)

    with st.form("questions_form"):
        answers = {}

        for i, question in enumerate(questions):
            q_type     = question['type']
            type_badge = '<span class="q-type-mcq">MCQ</span>' if q_type == 'mcq' else '<span class="q-type-open">Open Ended</span>'

            st.markdown(f"""
            <div class="q-card">
                <div class="q-header">
                    <span class="q-num">{i + 1}</span>
                    <span style="font-size:0.75rem;color:#94a3b8;font-weight:500;">
                        Question {i + 1} of {len(questions)}
                    </span>
                    {type_badge}
                </div>
                <div class="q-text">{question['question']}</div>
            </div>
            """, unsafe_allow_html=True)

            if q_type == 'mcq' and question.get('options'):
                answer = st.radio(
                    "Select the correct answer:",
                    question['options'],
                    key=f"q_{i}",
                    label_visibility="collapsed"
                )
                answers[i] = answer
            else:
                answer = st.text_area(
                    "Your answer:",
                    key=f"q_{i}",
                    height=120,
                    label_visibility="collapsed",
                    placeholder="Enter your detailed answer here..."
                )
                answers[i] = answer

            if i < len(questions) - 1:
                st.markdown("<hr style='border:none;border-top:1px solid #f1f5f9;margin:0.25rem 0;'>",
                            unsafe_allow_html=True)

        submitted = st.form_submit_button("📤 Submit Answers", use_container_width=True, type="primary")

        if submitted:
            st.session_state.answers = answers
            evaluate_answers()
            st.rerun()

def evaluate_answers():
    """Evaluate user answers and update state."""
    from src.llm_service import LLMService
    
    # Check Ollama connection first
    try:
        import httpx
        response = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        ollama_available = response.status_code == 200
    except:
        ollama_available = False
    
    if not ollama_available:
        st.error("⚠️ Cannot connect to Ollama")
        st.warning("""
        **Ollama is not running.** Please start Ollama to enable AI evaluation.
        
        To start Ollama:
        1. Open a terminal
        2. Run: `ollama serve`
        
        Or ensure Ollama is running in the background.
        """)
        st.info("Your answers have been saved. Restart after starting Ollama to get AI evaluation.")
        return
    
    with st.spinner("Evaluating your answers..."):
        llm_service = LLMService()
        questions = st.session_state.questions
        answers = st.session_state.answers
        state = st.session_state.learning_state
        
        verification_results = []
        
        for i, question in enumerate(questions):
            user_answer = answers.get(i, "")
            
            if not user_answer or user_answer.strip() == "":
                # Empty answer
                result = {
                    "question_id": question.get('question_id', f'q_{i}'),
                    "question": question['question'],
                    "learner_answer": user_answer,
                    "score": 0.0,
                    "feedback": "No answer provided",
                    "scoring_details": {"score": 0.0, "feedback": "No answer provided", "reasoning": "Empty response"}
                }
                verification_results.append(result)
                continue
            
            if question['type'] == 'mcq':
                correct_answer = question.get('correct_answer', '')
                # Check if user answer starts with correct letter
                is_correct = user_answer.strip()[0].upper() == correct_answer.strip()[0].upper()
                
                score = 1.0 if is_correct else 0.0
                feedback = "Correct!" if is_correct else f"Incorrect. The correct answer is {correct_answer}"
                
                scoring = {
                    "score": score,
                    "feedback": feedback,
                    "reasoning": feedback
                }
            else:
                # Open-ended - evaluate with LLM
                try:
                    expected_concepts = question.get('expected_concepts', [])
                    scoring = asyncio.run(llm_service.score_answer(
                        question['question'],
                        user_answer,
                        expected_concepts
                    ))
                except Exception as e:
                    logger.error(f"Error scoring answer: {e}")
                    # Check if it's an Ollama connection error
                    if "10061" in str(e) or "Connection" in str(e):
                        st.warning("⚠️ Cannot connect to Ollama. Please ensure Ollama is running.")
                        scoring = {
                            "score": 0.7,
                            "feedback": "Your answer shows understanding. (Note: AI evaluation unavailable - Ollama not running)",
                            "reasoning": "Ollama service not available. Please start Ollama to enable AI evaluation."
                        }
                    else:
                        # Other errors - fallback scoring
                        scoring = {
                            "score": 0.5,
                            "feedback": "Unable to evaluate automatically. Manual review needed.",
                            "reasoning": f"Evaluation error: {str(e)}"
                        }
            
            result = {
                "question_id": question.get('question_id', f'q_{i}'),
                "question": question['question'],
                "learner_answer": user_answer,
                "score": scoring['score'],
                "feedback": scoring['feedback'],
                "scoring_details": scoring
            }
            
            verification_results.append(result)
        
        # Calculate score
        total_score = sum(r['score'] for r in verification_results)
        avg_score = total_score / len(verification_results) if verification_results else 0
        score_percentage = avg_score * 100
        
        st.session_state.learning_state['verification_results'] = verification_results
        st.session_state.learning_state['score_percentage'] = score_percentage
        st.session_state.learning_state['meets_threshold'] = score_percentage >= 70.0
        
        st.session_state.stage = 'results'

def render_results():
    """Render assessment results."""
    state           = st.session_state.learning_state
    score           = state.get('score_percentage', 0)
    meets_threshold = state.get('meets_threshold', False)
    results         = state.get('verification_results', [])

    # Determine colour class for the big score number
    score_cls   = 'pass' if meets_threshold else ('warn' if score >= 50 else 'fail')
    verdict_cls = 'pass' if meets_threshold else 'fail'
    verdict_txt = '✅ Checkpoint Passed' if meets_threshold else '📚 More Practice Needed'

    # ── Score hero + summary metrics ──────────────────────────────────────────
    col_score, col_info = st.columns([1, 2])

    with col_score:
        st.markdown(f"""
        <div class="score-hero">
            <div class="score-sub">Your Score</div>
            <div class="score-num {score_cls}">{score:.0f}<span style="font-size:2.5rem;font-weight:700;">%</span></div>
            <div><span class="verdict {verdict_cls}">{verdict_txt}</span></div>
        </div>
        """, unsafe_allow_html=True)

    with col_info:
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Passing Threshold", "70%")
        with m2:
            st.metric("Questions Answered", len(results))

        if meets_threshold:
            st.markdown("""
            <div class="success-box" style="margin-top:0.75rem;">
                <strong>Excellent work!</strong> You've mastered this checkpoint and are ready to progress.
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="warning-box" style="margin-top:0.75rem;">
                <strong>Keep going!</strong> Review the simplified explanations below and retry — you've got this!
            </div>""", unsafe_allow_html=True)

    # ── Detailed per-question feedback ────────────────────────────────────────
    st.markdown('<div style="font-size:1.1rem;font-weight:700;color:#0f172a;margin:1.75rem 0 0.75rem;">📋 Detailed Feedback</div>',
                unsafe_allow_html=True)

    for i, result in enumerate(results, 1):
        score_pct = result['score'] * 100
        pill_cls  = 'rs-pass' if score_pct >= 70 else ('rs-mid' if score_pct >= 40 else 'rs-fail')
        label     = result['question'][:65] + ('…' if len(result['question']) > 65 else '')

        with st.expander(f"Q{i}: {label}", expanded=False):
            st.markdown(f'<span class="r-score {pill_cls}">{score_pct:.0f}%</span>', unsafe_allow_html=True)
            st.markdown(f"**Question:** {result['question']}")
            st.markdown(f"**Your Answer:** {result['learner_answer']}")
            st.markdown(f"**Feedback:** {result['feedback']}")

    # ── Feynman teaching on failure ───────────────────────────────────────────
    if not meets_threshold:
        render_feynman_teaching()

    # ── Navigation ────────────────────────────────────────────────────────────
    st.markdown("<hr style='border:none;border-top:1px solid #e2e8f0;margin:1.5rem 0;'>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        if not meets_threshold:
            if st.button("🔄 Retry Checkpoint", use_container_width=True):
                reset_checkpoint()
                st.rerun()

    with col2:
        if meets_threshold or st.session_state.learning_state.get('feynman_retry_count', 0) >= 3:
            if st.button("Next Checkpoint →", use_container_width=True, type="primary"):
                proceed_to_next_checkpoint()
                st.rerun()

def render_feynman_teaching():
    """Render Feynman teaching explanations."""
    st.markdown("""
    <div class="info-box" style="margin:1.5rem 0 1rem;">
        <strong>🧪 Feynman Technique</strong>
        <span style="font-weight:400;"> — Let's break down the concepts you found challenging into simple, clear explanations.</span>
    </div>
    """, unsafe_allow_html=True)

    from src.feynman_teaching import get_feynman_teacher
    
    feynman_teacher = get_feynman_teacher()
    state = st.session_state.learning_state
    
    # Identify knowledge gaps
    verification_results = state.get('verification_results', [])
    knowledge_gaps = feynman_teacher.identify_knowledge_gaps(verification_results)
    
    if knowledge_gaps:
        # Generate explanations
        with st.spinner("Generating simplified explanations..."):
            context_chunks = state.get('processed_context', [])
            
            # Check if Ollama is available before trying
            try:
                # Use run_coroutine_threadsafe or just run in new event loop
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # Streamlit already has a running loop
                        import nest_asyncio
                        nest_asyncio.apply()
                except:
                    pass
                
                explanations = asyncio.run(
                    feynman_teacher.generate_all_explanations(knowledge_gaps, context_chunks)
                )
            except Exception as e:
                logger.error(f"Error generating Feynman explanations: {e}")
                if "10061" in str(e) or "Connection" in str(e):
                    st.error("⚠️ Cannot generate explanations. Please ensure Ollama is running.")
                    st.info("Start Ollama with: `ollama serve` in a terminal")
                    explanations = []
                else:
                    st.error(f"Error generating explanations: {e}")
                    explanations = []
        
        # Display explanations
        for i, exp in enumerate(explanations, 1):
            with st.expander(f"💡 Concept {i}: {exp['concept']}  —  {exp['score']*100:.0f}% score", expanded=True):
                col_a, col_b = st.columns([1, 3])
                with col_a:
                    st.markdown(f"**Score:** `{exp['score']*100:.0f}%`")
                with col_b:
                    st.markdown(f"**Question:** {exp['question']}")
                st.markdown(f"**Your Answer:** {exp['user_answer'][:120]}{'…' if len(exp['user_answer']) > 120 else ''}")
                st.markdown("<hr style='border:none;border-top:1px solid #e2e8f0;margin:0.75rem 0;'>",
                            unsafe_allow_html=True)
                st.markdown("**Simplified Explanation:**")
                st.markdown(exp['simplified_explanation'])

def reset_checkpoint():
    """Reset current checkpoint for retry."""
    # Get retry count before resetting
    retry_count = st.session_state.learning_state.get('feynman_retry_count', 0) if st.session_state.learning_state else 0
    
    st.session_state.learning_state = None
    st.session_state.questions = []
    st.session_state.answers = {}
    st.session_state.study_materials_ready = False
    st.session_state.stage = 'learning'
    
    # Store retry count for next iteration
    if retry_count < 3:
        if 'feynman_retry_metadata' not in st.session_state:
            st.session_state.feynman_retry_metadata = {}
        st.session_state.feynman_retry_metadata['retry_count'] = retry_count + 1

def proceed_to_next_checkpoint():
    """Move to next checkpoint."""
    current_idx = st.session_state.current_checkpoint_index
    st.session_state.completed_checkpoints.append(
        st.session_state.selected_path['checkpoints'][current_idx]['id']
    )
    st.session_state.current_checkpoint_index += 1
    st.session_state.learning_state = None
    st.session_state.questions = []
    st.session_state.answers = {}
    st.session_state.study_materials_ready = False
    st.session_state.stage = 'learning'

def render_completion():
    """Render completion screen."""
    st.balloons()

    path      = st.session_state.selected_path
    completed = len(st.session_state.completed_checkpoints)
    total     = len(path['checkpoints'])
    icon      = get_path_icon(path['title'])

    # Trophy hero banner
    st.markdown(f"""
    <div class="completion-hero">
        <div style="font-size:5rem;margin-bottom:0.75rem;">🏆</div>
        <h2>Learning Path Complete!</h2>
        <p>Congratulations! You've mastered <strong>{icon} {path['title']}</strong></p>
    </div>
    """, unsafe_allow_html=True)

    # Stats grid
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-v">{completed}</div>
            <div class="stat-l">Checkpoints Completed</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-v">100%</div>
            <div class="stat-l">Path Progress</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-v">✓</div>
            <div class="stat-l">Path Mastered</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        if st.button("🚀 Start New Learning Path", use_container_width=True, type="primary"):
            st.session_state.stage = 'select_path'
            st.session_state.selected_path = None
            st.session_state.current_checkpoint_index = 0
            st.session_state.uploaded_files = []
            st.session_state.learning_state = None
            st.session_state.questions = []
            st.session_state.answers = {}
            st.session_state.completed_checkpoints = []
            st.rerun()

def main():
    """Main application entry point."""
    init_session_state()
    render_header()
    
    # Route to appropriate stage
    if st.session_state.stage == 'select_path':
        render_path_selection()
    elif st.session_state.stage == 'upload_files':
        render_file_upload()
    elif st.session_state.stage == 'study':
        render_study_materials()
    elif st.session_state.stage == 'learning':
        render_learning_session()
    elif st.session_state.stage == 'results':
        render_results()

if __name__ == "__main__":
    main()
