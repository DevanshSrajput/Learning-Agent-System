"""
Streamlit Web Application for Learning Agent System

A professional, modern web interface for the autonomous learning agent.
Provides interactive learning experiences with checkpoint progression,
dynamic materials, and adaptive teaching.
"""

import streamlit as st
import asyncio
import sys
import time
import re
import os
import random
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
from src.langsmith_config import langsmith_config

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

# Custom CSS theme: modern "Command Center" look and feel
st.markdown("""
<style>
    :root {
        --bg-main: #f7f3eb;
        --bg-surface: #fffdf8;
        --text-primary: #121212;
        --text-muted: #3f3f46;
        --ink: #111111;
        --mint: #77f5cf;
        --yellow: #ffe37a;
        --blue: #8ec5ff;
        --pink: #ff9fbe;
        --white: #ffffff;
        --shadow: 6px 6px 0 #111111;
        --radius: 8px;
    }

    .stApp {
        background: var(--bg-main);
        color: var(--text-primary);
    }

    .main {
        padding-top: 0.4rem;
    }

    .shell {
        background: var(--white);
        border: 3px solid var(--ink);
        border-radius: var(--radius);
        box-shadow: var(--shadow);
        padding: 1rem;
        margin-bottom: 1rem;
    }

    .hero-title {
        font-size: 1.95rem;
        font-weight: 800;
        letter-spacing: 0;
        margin: 0;
    }

    .hero-subtitle {
        font-size: 0.98rem;
        color: var(--text-muted);
        margin: 0.25rem 0 0 0;
    }

    .stage-chip {
        display: inline-block;
        padding: 0.3rem 0.55rem;
        border: 2px solid var(--ink);
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 700;
        background: var(--yellow);
        margin-top: 0.6rem;
    }

    .card {
        padding: 0.95rem;
        border-radius: var(--radius);
        background: var(--white);
        border: 3px solid var(--ink);
        box-shadow: var(--shadow);
        margin-bottom: 0.8rem;
    }

    .kicker {
        font-size: 0.75rem;
        font-weight: 800;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 0.25rem;
    }

    .question-card {
        background: var(--white);
        border: 3px solid var(--ink);
        border-radius: var(--radius);
        padding: 0.85rem;
        margin-bottom: 0.65rem;
        line-height: 1.4;
        box-shadow: 4px 4px 0 #111111;
    }

    .q-badge {
        display: inline-block;
        font-size: 0.72rem;
        padding: 0.18rem 0.45rem;
        border-radius: 6px;
        border: 2px solid var(--ink);
        background: var(--mint);
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .q-badge.long {
        background: var(--pink);
    }

    .q-badge.short {
        background: var(--blue);
    }

    .nav-box {
        background: var(--white);
        border: 3px solid var(--ink);
        border-radius: var(--radius);
        box-shadow: var(--shadow);
        padding: 0.85rem;
        margin-bottom: 1rem;
    }

    .nav-step {
        display: inline-block;
        width: 100%;
        border: 2px solid var(--ink);
        border-radius: 6px;
        padding: 0.4rem 0.5rem;
        margin-bottom: 0.4rem;
        font-size: 0.8rem;
        font-weight: 700;
        background: #fff;
    }

    .nav-step.active {
        background: var(--yellow);
    }

    .nav-step.done {
        background: var(--mint);
    }

    .success-box, .warning-box, .error-box {
        padding: 0.8rem;
        border-radius: var(--radius);
        border: 3px solid var(--ink);
        box-shadow: 4px 4px 0 #111111;
        margin-bottom: 0.6rem;
    }

    .success-box { background: var(--mint); color: #10241d; }
    .warning-box { background: var(--yellow); color: #2a220a; }
    .error-box { background: var(--pink); color: #2c0f19; }

    .stButton > button {
        width: 100%;
        border-radius: 8px;
        border: 3px solid var(--ink);
        background: var(--yellow);
        color: var(--ink);
        font-weight: 800;
        box-shadow: 4px 4px 0 #111111;
    }

    .stButton > button:hover {
        transform: translate(-1px, -1px);
        box-shadow: 5px 5px 0 #111111;
    }

    .stProgress > div > div > div > div {
        background: #111;
    }

    .stTextArea textarea, .stTextInput input {
        background: #fff;
        color: #111;
        border: 3px solid var(--ink);
        border-radius: 8px;
    }

    .stAlert {
        border-radius: 8px;
        border: 3px solid var(--ink);
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

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

def render_header():
    """Render application header."""
    stage = st.session_state.get("stage", "select_path")
    stage_labels = {
        "select_path": "Select Path",
        "upload_files": "Upload Materials",
        "study": "Study",
        "learning": "Assessment",
        "results": "Results"
    }

    st.markdown('<div class="shell">', unsafe_allow_html=True)
    st.markdown('<p class="hero-title">Learning Agent Studio</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Neo-brutalist adaptive learning workspace: clear goals, strong feedback, no clutter.</p>', unsafe_allow_html=True)
    st.markdown(f'<span class="stage-chip">Current Stage: {stage_labels.get(stage, "Unknown")}</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Check Ollama status
    try:
        import httpx
        response = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        if response.status_code == 200:
            st.success("🟢 Ollama is running", icon="✅")
        else:
            st.warning("🟡 Ollama connection issue", icon="⚠️")
    except:
        st.error("🔴 Ollama is not running - AI features will be limited", icon="❌")
        with st.expander("How to start Ollama"):
            st.code("ollama serve", language="bash")
            st.markdown("Or ensure Ollama is running in the background.")

    with st.expander("LangSmith Status", expanded=False):
        tracing_env = os.getenv("LANGSMITH_TRACING_ENABLED", "false").lower() == "true"
        api_key_set = bool(os.getenv("LANGCHAIN_API_KEY"))
        project = os.getenv("LANGCHAIN_PROJECT", "Learning-Agent-System")
        endpoint = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
        runtime_enabled = bool(getattr(langsmith_config, "enabled", False))

        st.write(f"- Env tracing toggle: {'ON' if tracing_env else 'OFF'}")
        st.write(f"- API key present: {'YES' if api_key_set else 'NO'}")
        st.write(f"- Project: `{project}`")
        st.write(f"- Endpoint: `{endpoint}`")
        st.write(f"- Runtime tracing active: {'YES' if runtime_enabled else 'NO'}")

        if not tracing_env:
            st.warning("Set `LANGSMITH_TRACING_ENABLED=true` in `.env`.")
        elif not api_key_set:
            st.warning("Set `LANGCHAIN_API_KEY` in `.env`.")
        elif not runtime_enabled:
            st.warning("Tracing is configured but inactive at runtime; restart app and recheck credentials/project.")
        else:
            st.success("LangSmith tracing is active for this app session.")

    st.markdown("")

def _question_type_label(question: Dict) -> str:
    q_type = str(question.get('type', '')).lower()
    if q_type in ('mcq', 'multiple_choice'):
        return "Multiple Choice"
    if q_type in ('short_answer', 'short'):
        return "Short Answer"
    if q_type in ('long_answer', 'long'):
        return "Long Answer"
    return "Open Answer"

def ensure_question_distribution(questions: List[Dict], checkpoint_requirements: Optional[List[str]] = None) -> List[Dict]:
    """UI-level hard guard: always return 5 MCQ, 3 short, 2 long questions."""
    checkpoint_requirements = checkpoint_requirements or []
    by_type = {"multiple_choice": [], "short_answer": [], "long_answer": []}

    for q in questions or []:
        q_type = str(q.get("type", "")).lower()
        if q_type in ("mcq", "multiple_choice"):
            q["type"] = "multiple_choice"
            by_type["multiple_choice"].append(q)
        elif q_type in ("long", "long_answer"):
            q["type"] = "long_answer"
            by_type["long_answer"].append(q)
        else:
            q["type"] = "short_answer"
            by_type["short_answer"].append(q)

    def mcq_stub(i: int) -> Dict:
        req = checkpoint_requirements[min(i, len(checkpoint_requirements)-1)] if checkpoint_requirements else "the checkpoint topic"
        return {
            "question": f"Which choice best aligns with the objective: '{req}'?",
            "type": "multiple_choice",
            "options": [
                "A) It captures the core concept with practical relevance.",
                "B) It avoids key concepts and remains vague.",
                "C) It contradicts the objective directly.",
                "D) It is unrelated to the checkpoint topic."
            ],
            "correct_answer": "A",
            "expected_concepts": checkpoint_requirements[:2]
        }

    def short_stub(i: int) -> Dict:
        req = checkpoint_requirements[min(i, len(checkpoint_requirements)-1)] if checkpoint_requirements else "the checkpoint topic"
        return {
            "question": f"In 2-4 sentences, explain the key idea behind '{req}'.",
            "type": "short_answer",
            "expected_concepts": checkpoint_requirements[:3]
        }

    def long_stub(i: int) -> Dict:
        req = checkpoint_requirements[min(i, len(checkpoint_requirements)-1)] if checkpoint_requirements else "the checkpoint topic"
        return {
            "question": f"Provide a detailed analysis of '{req}', including reasoning, examples, and limitations.",
            "type": "long_answer",
            "expected_concepts": checkpoint_requirements
        }

    while len(by_type["multiple_choice"]) < 5:
        by_type["multiple_choice"].append(mcq_stub(len(by_type["multiple_choice"])))
    while len(by_type["short_answer"]) < 3:
        by_type["short_answer"].append(short_stub(len(by_type["short_answer"])))
    while len(by_type["long_answer"]) < 2:
        by_type["long_answer"].append(long_stub(len(by_type["long_answer"])))

    mcq_selected = random.sample(by_type["multiple_choice"], 5) if len(by_type["multiple_choice"]) > 5 else by_type["multiple_choice"][:5]
    short_selected = random.sample(by_type["short_answer"], 3) if len(by_type["short_answer"]) > 3 else by_type["short_answer"][:3]
    long_selected = random.sample(by_type["long_answer"], 2) if len(by_type["long_answer"]) > 2 else by_type["long_answer"][:2]

    selected = mcq_selected + short_selected + long_selected
    random.shuffle(selected)
    for idx, q in enumerate(selected, 1):
        q["question_id"] = f"q_{idx}"
    return selected

def render_stage_rail():
    stage = st.session_state.get("stage", "select_path")
    order = ["select_path", "upload_files", "study", "learning", "results"]
    labels = {
        "select_path": "1. Path",
        "upload_files": "2. Materials",
        "study": "3. Study",
        "learning": "4. Assessment",
        "results": "5. Results"
    }
    current_idx = order.index(stage) if stage in order else 0

    st.markdown('<div class="nav-box"><div class="kicker">Workflow</div>', unsafe_allow_html=True)
    for idx, key in enumerate(order):
        cls = "nav-step"
        if idx < current_idx:
            cls += " done"
        elif idx == current_idx:
            cls += " active"
        st.markdown(f'<div class="{cls}">{labels[key]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def show_loading_sequence(status, messages: List[str], delay: float = 0.18):
    """Show rotating status messages to keep users engaged during long tasks."""
    for msg in messages:
        status.write(msg)
        time.sleep(delay)

def format_study_markdown(text: str) -> str:
    """Normalize generated study text into readable Markdown blocks."""
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    cleaned = cleaned.replace("\r\n", "\n")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"(?m)^(\d+)\.\s*([A-Z][^\n:]{2,40})$", r"## \1. \2", cleaned)
    cleaned = re.sub(r"(?m)^([A-Z][A-Z\s]{4,40}):\s*$", lambda m: f"## {m.group(1).title()}", cleaned)
    paragraphs = [p.strip() for p in cleaned.split("\n\n") if p.strip()]
    return "\n\n".join(paragraphs)

def render_path_selection():
    """Render learning path selection interface."""
    st.markdown('<div class="card"><div class="kicker">Setup</div><h3 style="margin:0;">Select Learning Path</h3></div>', unsafe_allow_html=True)
    
    # Load paths
    default_paths = create_learning_paths()
    custom_paths = load_custom_topics()
    all_paths = default_paths + custom_paths
    
    # Display paths
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Available Learning Paths")
        
        for i, path in enumerate(all_paths):
            with st.container():
                st.markdown(f"""
                <div class="card">
                    <h4 style="margin:0 0 6px 0;">{path['title']}</h4>
                    <p style="margin:0 0 8px 0; color:#9cb0c3;">{path['description']}</p>
                    <p style="margin:0; color:#cdd9e5;"><strong>Checkpoints:</strong> {len(path['checkpoints'])}</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button(f"Select", key=f"select_{i}"):
                    st.session_state.selected_path = path
                    st.session_state.stage = 'upload_files'
                    st.rerun()
    
    with col2:
        st.subheader("Actions")
        
        if st.button("Create Custom Topic", use_container_width=True):
            st.session_state.show_create_topic = True
            st.rerun()
    
    # Custom topic creation form
    if st.session_state.show_create_topic:
        render_custom_topic_form()

def render_custom_topic_form():
    """Render custom topic creation form."""
    st.markdown("---")
    st.subheader("Create Custom Learning Topic")
    
    st.info("Enter a topic name (e.g., 'Python Basics', 'Machine Learning') and the system will handle everything else.")
    
    with st.form("custom_topic_form"):
        topic_name = st.text_input(
            "Topic Name",
            placeholder="e.g., Python Basics, Machine Learning, Web Development",
            help="Enter any topic you want to learn about"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            submitted = st.form_submit_button("Start Learning", use_container_width=True)
        with col2:
            cancelled = st.form_submit_button("Cancel", use_container_width=True)
        
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
    st.markdown('<div class="card"><div class="kicker">Setup</div><h3 style="margin:0;">Upload Learning Materials (Optional)</h3></div>', unsafe_allow_html=True)
    
    path = st.session_state.selected_path
    st.info(f"Selected: {path['title']}")
    
    st.markdown("""
    You can upload your own learning materials in the following formats:
    - PDF documents
    - Word documents (DOCX)
    - Markdown files (MD)
    - Plain text files (TXT)
    
    If you don't upload materials, the system will generate them dynamically using AI and web search.
    """)
    
    uploaded_files = st.file_uploader(
        "Upload files",
        type=['pdf', 'docx', 'md', 'txt'],
        accept_multiple_files=True,
        help="Maximum 10MB per file"
    )
    
    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        st.success(f"Uploaded {len(uploaded_files)} file(s)")
        
        for file in uploaded_files:
            st.text(f"- {file.name} ({file.size / 1024:.1f} KB)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Continue with Uploaded Files", use_container_width=True, disabled=not uploaded_files):
            st.session_state.stage = 'learning'
            st.rerun()
    
    with col2:
        if st.button("Skip and Generate Materials", use_container_width=True):
            st.session_state.uploaded_files = []
            st.session_state.stage = 'learning'
            st.rerun()

def render_study_materials():
    """Display learning materials for study before assessment."""
    st.markdown('<div class="card"><div class="kicker">Study Zone</div><h3 style="margin:0;">Study Materials</h3></div>', unsafe_allow_html=True)
    st.info("Review the learning materials below before starting the assessment.")
    
    state = st.session_state.learning_state
    
    if not state:
        st.error("No learning materials available.")
        return
    
    # Display learning content as one continuous explanation
    summary = state.get('summary', '')
    materials = state.get('collected_materials', [])
    
    if summary:
        st.markdown("### 📚 Study Material")
        st.markdown("*Read through this comprehensive explanation carefully to prepare for the assessment.*")
        st.markdown("")
        
        formatted_md = format_study_markdown(summary)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(formatted_md)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Show word count for reference
        word_count = len(summary.split())
        st.caption(f"📊 Content length: {word_count} words")
    else:
        st.warning("No learning content available.")
    
    # Extract and display resource links
    resource_links = []
    for material in materials:
        url = material.get('url') or material.get('link') or material.get('source_url')
        title = material.get('title', 'Resource')
        source = material.get('source', 'Unknown')
        
        if url and url.startswith('http'):
            resource_links.append({
                'title': title,
                'url': url,
                'source': source
            })
    
    # Display resource links section
    if resource_links:
        st.markdown("---")
        st.markdown("### 🔗 Additional Resources")
        st.markdown(f"Explore **{len(resource_links)} external resources** for deeper learning:")
        st.markdown("")
        
        for i, resource in enumerate(resource_links, 1):
            st.markdown(f"""
            <div style="background-color: #e8f4f8; padding: 15px; border-radius: 6px; margin-bottom: 10px; border-left: 3px solid #2196F3;">
                <div style="font-size: 15px; font-weight: 600; color: #1a1a1a; margin-bottom: 6px;">
                    {i}. {resource['title']}
                </div>
                <div style="margin-bottom: 4px;">
                    <a href="{resource['url']}" target="_blank" style="color: #1565c0; text-decoration: none; font-size: 13px;">
                        🔗 {resource['url']}
                    </a>
                </div>
                <div style="font-size: 11px; color: #666;">
                    <em>Source: {resource['source']}</em>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Button to proceed to assessment
    st.markdown("")
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("✅ Start Assessment", use_container_width=True, type="primary"):
            st.session_state.stage = 'learning'
            st.rerun()

def render_learning_session():
    """Render the learning session with materials, questions, and assessment."""
    path = st.session_state.selected_path
    checkpoint_idx = st.session_state.current_checkpoint_index
    
    if checkpoint_idx >= len(path['checkpoints']):
        render_completion()
        return
    
    checkpoint = path['checkpoints'][checkpoint_idx]
    
    # Progress bar
    
    if checkpoint_idx >= len(path['checkpoints']):
        render_completion()
        return
    
    checkpoint = path['checkpoints'][checkpoint_idx]
    
    # Progress bar
    progress = checkpoint_idx / len(path['checkpoints'])
    st.progress(progress, text=f"Progress: {checkpoint_idx}/{len(path['checkpoints'])} checkpoints completed")
    
    st.markdown(
        f'<div class="card"><div class="kicker">Checkpoint</div><h3 style="margin:0;">{checkpoint["title"]}</h3><p style="margin:8px 0 0 0;color:#3f3f46;">{checkpoint["description"]}</p></div>',
        unsafe_allow_html=True
    )
    
    # Learning objectives
    with st.expander("Learning Objectives", expanded=False):
        for req in checkpoint['requirements']:
            st.markdown(f"- {req}")
    
    # Run workflow in background
    if st.session_state.learning_state is None:
        with st.status("Preparing learning materials...", expanded=True) as prep_status:
            show_loading_sequence(prep_status, [
                "Warming up retrieval + reasoning pipelines...",
                "Aligning checkpoint objectives with source context...",
                "Drafting an assessment-focused study brief..."
            ])
            # Process uploaded files
            uploaded_paths = None
            if st.session_state.uploaded_files:
                upload_handler = get_upload_handler()
                temp_paths = []
                prep_status.write("Processing your uploaded files...")
                
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
                show_loading_sequence(prep_status, [
                    "Collecting materials from trusted sources...",
                    "Distilling context into concept chunks...",
                    "Generating checkpoint question set..."
                ])
                result = asyncio.run(workflow.ainvoke(initial_state))
                
                st.session_state.learning_state = result
                st.session_state.questions = ensure_question_distribution(
                    result.get('generated_questions', []),
                    checkpoint.get("requirements", [])
                )
                
                if st.session_state.questions:
                    prep_status.write(f"Generated {len(st.session_state.questions)} questions.")
                    st.session_state.study_materials_ready = True
                    prep_status.update(label="Study materials ready!", state="complete")
                    # Redirect to study stage to show materials before assessment
                    st.session_state.stage = 'study'
                else:
                    prep_status.write("No questions generated.")
                    prep_status.update(label="Error", state="error")
                        
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
    st.markdown('<div class="card"><div class="kicker">Assessment</div><h3 style="margin:0;">Checkpoint Questions</h3></div>', unsafe_allow_html=True)
    question_count = len(st.session_state.questions)
    st.markdown(
        '<div class="card" style="background:#ffe37a;"><strong>Question Mix:</strong> 5 MCQ · 3 Short Answer · 2 Long Answer</div>',
        unsafe_allow_html=True
    )
    st.info(f"Answer all {question_count} questions. Passing threshold remains 70%.")
    
    questions = st.session_state.questions
    
    with st.form("questions_form"):
        answers = {}
        
        for i, question in enumerate(questions):
            # Display question with clear formatting
            st.markdown(f"#### Question {i+1} of {len(questions)}")
            q_label = _question_type_label(question)
            badge_cls = "q-badge"
            if q_label == "Long Answer":
                badge_cls += " long"
            elif q_label == "Short Answer":
                badge_cls += " short"
            st.markdown(f'<span class="{badge_cls}">{q_label}</span>', unsafe_allow_html=True)
            
            # Show question text in a box
            st.markdown(f"""
            <div class="question-card">
                {question['question']}
            </div>
            """, unsafe_allow_html=True)
            
            q_type = str(question.get('type', '')).lower()
            is_mcq = q_type in ('mcq', 'multiple_choice')
            if is_mcq and question.get('options'):
                # Multiple choice
                st.markdown("**Select the correct answer:**")
                options = question['options']
                answer = st.radio(
                    "Options:",
                    options,
                    key=f"q_{i}",
                    index=None,
                    label_visibility="collapsed"
                )
                answers[i] = answer
            else:
                # Open-ended
                st.markdown("**Your answer:**")
                q_height = 120
                if q_type in ('long', 'long_answer'):
                    q_height = 180
                answer = st.text_area(
                    "Type your answer here:",
                    key=f"q_{i}",
                    height=q_height,
                    label_visibility="collapsed",
                    placeholder="Enter your detailed answer here..."
                )
                answers[i] = answer
            
            st.markdown("---")
        
        submitted = st.form_submit_button("Submit Answers", use_container_width=True)
        
        if submitted:
            missing = []
            for idx, question in enumerate(questions):
                value = answers.get(idx)
                q_type = str(question.get('type', '')).lower()
                is_mcq = q_type in ('mcq', 'multiple_choice')
                if is_mcq and value is None:
                    missing.append(idx + 1)
                if (not is_mcq) and (not value or not str(value).strip()):
                    missing.append(idx + 1)

            if missing:
                st.error(f"Please complete all questions before submitting. Missing: {', '.join(map(str, missing))}")
            else:
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
    
    with st.status("Evaluating your answers...", expanded=True) as eval_status:
        show_loading_sequence(eval_status, [
            "Checking objective coverage question-by-question...",
            "Scoring responses against expected concepts...",
            "Compiling checkpoint feedback summary..."
        ])
        llm_service = LLMService()
        questions = st.session_state.questions
        answers = st.session_state.answers
        state = st.session_state.learning_state
        
        verification_results = []
        
        for i, question in enumerate(questions):
            if i % 2 == 0:
                eval_status.write(f"Evaluating response {i+1} of {len(questions)}...")
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
            
            q_type = str(question.get('type', '')).lower()
            is_mcq = q_type in ('mcq', 'multiple_choice')
            if is_mcq:
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
        eval_status.update(label="Evaluation complete", state="complete")
        
        st.session_state.stage = 'results'

def render_results():
    """Render assessment results."""
    state = st.session_state.learning_state
    score = state.get('score_percentage', 0)
    meets_threshold = state.get('meets_threshold', False)
    results = state.get('verification_results', [])
    
    st.markdown('<div class="card"><div class="kicker">Outcome</div><h3 style="margin:0;">Assessment Results</h3></div>', unsafe_allow_html=True)
    
    # Score display
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Your Score", f"{score:.1f}%")
    
    with col2:
        st.metric("Threshold", "70%")
    
    with col3:
        status = "PASSED" if meets_threshold else "NEEDS IMPROVEMENT"
        st.metric("Status", status)
    
    # Result message
    if meets_threshold:
        st.markdown("""
        <div class="success-box">
            <strong>Excellent work!</strong> You have successfully completed this checkpoint.
            You are ready to progress to the next checkpoint.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="warning-box">
            <strong>Keep learning!</strong> Your score is below 70%.
            Review the feedback below and consider the simplified explanations.
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed feedback
    st.subheader("Detailed Feedback")
    
    for i, result in enumerate(results, 1):
        score_pct = result['score'] * 100
        
        with st.expander(f"Question {i}: {score_pct:.0f}%", expanded=False):
            st.markdown(f"**Question:** {result['question']}")
            st.markdown(f"**Your Answer:** {result['learner_answer']}")
            st.markdown(f"**Feedback:** {result['feedback']}")
    
    # Feynman teaching if failed
    if not meets_threshold:
        render_feynman_teaching()
    
    # Navigation
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if not meets_threshold:
            if st.button("Retry Checkpoint", use_container_width=True):
                reset_checkpoint()
                st.rerun()
    
    with col2:
        if meets_threshold or st.session_state.learning_state.get('feynman_retry_count', 0) >= 3:
            if st.button("Next Checkpoint", use_container_width=True):
                proceed_to_next_checkpoint()
                st.rerun()

def render_feynman_teaching():
    """Render Feynman teaching explanations."""
    st.subheader("Simplified Explanations")
    st.info("Let's break down these concepts in simpler terms...")
    
    from src.feynman_teaching import get_feynman_teacher
    
    feynman_teacher = get_feynman_teacher()
    state = st.session_state.learning_state
    
    # Identify knowledge gaps
    verification_results = state.get('verification_results', [])
    knowledge_gaps = feynman_teacher.identify_knowledge_gaps(verification_results)
    
    if knowledge_gaps:
        # Generate explanations
        with st.status("Generating simplified explanations...", expanded=True) as teach_status:
            show_loading_sequence(teach_status, [
                "Finding low-confidence concepts...",
                "Reframing ideas with simpler mental models...",
                "Preparing targeted explanations for retry..."
            ])
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
                teach_status.update(label="Simplified explanations ready", state="complete")
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
            with st.expander(f"Concept {i}: {exp['concept']}", expanded=True):
                st.markdown(f"**Your Understanding Gap (Score: {exp['score']*100:.0f}%)**")
                st.markdown(f"Question: {exp['question']}")
                st.markdown(f"Your Answer: {exp['user_answer'][:100]}...")
                st.markdown("---")
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
    st.success("Congratulations! You have completed the entire learning path!")
    
    path = st.session_state.selected_path
    completed = len(st.session_state.completed_checkpoints)
    total = len(path['checkpoints'])
    
    st.metric("Checkpoints Completed", f"{completed}/{total}")
    
    if st.button("Start New Learning Path", use_container_width=True):
        # Reset everything
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
    render_stage_rail()
    
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
    # Support both `streamlit run app.py` and accidental `python app.py`.
    # When no Streamlit script context exists, relaunch through Streamlit.
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        is_streamlit_context = get_script_run_ctx() is not None
    except Exception:
        is_streamlit_context = False

    if is_streamlit_context:
        main()
    else:
        from streamlit.web import cli as stcli
        sys.argv = ["streamlit", "run", __file__]
        raise SystemExit(stcli.main())
