import io
import random
from datetime import datetime

import cv2
import pandas as pd
import streamlit as st
from PIL import Image

from ml_model import predict_behavior
from attendance import (
    save_attendance,
    get_attendance,
    get_student_attendance_summary
)
from database import (
    create_session,
    add_event,
    finish_session,
    save_student,
    get_students,
    get_sessions,
    get_events
)


# =========================================================
# PAGE CONFIG (must be the first Streamlit command)
# =========================================================

st.set_page_config(
    page_title="ClassSense AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# SESSION STATE DEFAULTS
# =========================================================

_STATE_DEFAULTS = {
    "authenticated": False,
    "active_session_id": None,
    "active_session_start": None,
    "live_max_people": 0,
    "live_max_phones": 0,
    "live_max_laptops": 0,
    "live_phone_events": 0,
    "last_session_summary": None,
}

for _key, _default in _STATE_DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _default


# =========================================================
# CONSTANTS
# =========================================================

DEMO_USERNAME = "teacher"
DEMO_PASSWORD = "classsense123"

CATEGORY_ORDER = ["Excellent", "Good", "Needs Attention"]

CATEGORY_LABELS = {
    "Excellent": "🟢 Excellent",
    "Good": "🔵 Good",
    "Needs Attention": "🟠 Needs Attention",
}

EVENT_TYPE_LABELS = {
    "PHONE_DETECTED": "📱 Possible phone detected",
    "IMAGE_ANALYSIS": "🖼️ Image analysis",
    "SESSION_STARTED": "▶️ Session started",
    "SESSION_ENDED": "🛑 Session ended",
}

STATUS_LABELS = {
    "Completed": "✅ Completed",
    "Active": "🟢 Active",
    "Not closed": "⚪ Not closed",
}

ATTENDANCE_LABELS = {
    "Present": "✅ Present",
    "Absent": "❌ Absent",
}

TEACHER_NOTE = (
    "ClassSense AI is a teacher-support analytics prototype. Its outputs are "
    "indicators for teacher review only. They are not marks, promotion "
    "decisions or disciplinary decisions."
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

[data-testid="stSidebar"] {
    background-color: #111827;
}

[data-testid="stSidebar"] * {
    color: white;
}

.hero {
    padding: 25px;
    border-radius: 18px;
    background: linear-gradient(135deg, #111827, #1e3a8a);
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 38px;
    margin-bottom: 5px;
    color: white;
}

.hero p {
    font-size: 16px;
    color: #dbeafe;
    margin: 0;
}

.card {
    padding: 20px;
    border-radius: 16px;
    background: #1f2937;
    color: #f9fafb;
    border: 1px solid #374151;
    box-shadow: 0 4px 15px rgba(0,0,0,0.15);
}

.status {
    padding: 10px 15px;
    border-radius: 10px;
    background: #dcfce7;
    color: #166534;
    font-weight: 600;
}

.section-title {
    font-size: 25px;
    font-weight: 700;
    margin-top: 10px;
    margin-bottom: 15px;
}

.kpi {
    padding: 16px 18px;
    border-radius: 14px;
    background: linear-gradient(135deg, #1f2937, #111827);
    border: 1px solid #374151;
    margin-bottom: 10px;
    min-height: 108px;
}

.kpi-label {
    font-size: 12px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #9ca3af;
    font-weight: 600;
}

.kpi-value {
    font-size: 27px;
    font-weight: 800;
    color: #f9fafb;
    line-height: 1.25;
    margin-top: 4px;
}

.kpi-note {
    font-size: 12px;
    color: #93c5fd;
    margin-top: 2px;
}

.footer {
    text-align: center;
    color: #6b7280;
    padding: 30px;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def format_ts(value, fmt="%d %b %Y, %H:%M:%S"):
    """Convert an ISO timestamp string into a readable one."""
    if not value:
        return "—"
    try:
        return datetime.fromisoformat(str(value)).strftime(fmt)
    except ValueError:
        return str(value)


def format_duration(seconds):
    """Convert seconds into text such as 1h 05m 09s."""
    try:
        seconds = int(seconds or 0)
    except (TypeError, ValueError):
        seconds = 0

    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def kpi_cards(items):
    """Show a row of summary cards. items = [(label, value, note), ...]"""
    columns = st.columns(len(items))
    for column, (label, value, note) in zip(columns, items):
        column.markdown(
            f'<div class="kpi">'
            f'<div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div>'
            f'<div class="kpi-note">{note}</div>'
            f'</div>',
            unsafe_allow_html=True
        )


def show_table(df, column_config=None):
    """Show a clean full-width table without the row index."""
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config=column_config
    )


def csv_button(label, df, filename, key):
    """Download button that exports a DataFrame as CSV."""
    st.download_button(
        label,
        df.to_csv(index=False).encode("utf-8"),
        filename,
        "text/csv",
        key=key
    )


def safe_predict(attendance, phone_events, interactions, activity_level):
    """Run the Random Forest model; return (category, error_text)."""
    try:
        category = predict_behavior(
            attendance,
            phone_events,
            interactions,
            activity_level
        )
        return str(category), None
    except Exception as error:
        return None, str(error)


@st.cache_resource(show_spinner="Loading YOLO model...")
def load_yolo_model():
    """Load YOLO once and reuse it for image analysis."""
    from ultralytics import YOLO
    return YOLO("yolo11n.pt")


def build_student_df(students):
    """Convert student rows from the database into a DataFrame."""
    rows = []
    for row in students:
        rows.append({
            "Student ID": row[0],
            "Attendance (%)": float(row[1] or 0),
            "Phone Events": row[2],
            "Interactions": row[3],
            "Activity (%)": float(row[4] or 0),
            "AI Category": row[5],
            "Updated": row[6],
        })
    return pd.DataFrame(rows)


def student_display(df):
    """Make a student DataFrame readable for on-screen tables."""
    view = df.copy()
    view["AI Category"] = view["AI Category"].map(
        lambda c: CATEGORY_LABELS.get(c, c)
    )
    view["Updated"] = view["Updated"].map(lambda v: format_ts(v))
    return view


STUDENT_COLUMN_CONFIG = {
    "Attendance (%)": st.column_config.ProgressColumn(
        "Attendance",
        format="%.0f%%",
        min_value=0,
        max_value=100
    ),
    "Activity (%)": st.column_config.ProgressColumn(
        "Activity",
        format="%.0f%%",
        min_value=0,
        max_value=100
    ),
}

ATTENDANCE_SUMMARY_CONFIG = {
    "Attendance %": st.column_config.ProgressColumn(
        "Attendance %",
        format="%.1f%%",
        min_value=0,
        max_value=100
    ),
}


def category_options(df):
    extras = [
        c for c in df["AI Category"].dropna().unique()
        if c not in CATEGORY_ORDER
    ]
    return CATEGORY_ORDER + extras


def build_attendance_summary_df(summary):
    rows = []
    for student_id, present, total in summary:
        present = present or 0
        percentage = (present / total * 100) if total else 0
        rows.append({
            "Student ID": student_id,
            "Present": present,
            "Total Sessions": total,
            "Attendance %": round(percentage, 1),
            "Teacher Review": "Below 75%" if percentage < 75 else "OK",
        })
    return pd.DataFrame(rows)


# ---------- Session helpers (used as button callbacks) ----------

def reset_live_stats():
    st.session_state.live_max_people = 0
    st.session_state.live_max_phones = 0
    st.session_state.live_max_laptops = 0
    st.session_state.live_phone_events = 0


def start_new_session(note="Classroom session started by teacher"):
    session_id = create_session()
    st.session_state.active_session_id = session_id
    st.session_state.active_session_start = datetime.now()
    st.session_state.last_session_summary = None
    reset_live_stats()
    add_event(session_id, "SESSION_STARTED", note)
    return session_id


def end_current_session():
    session_id = st.session_state.active_session_id

    if session_id is None:
        return

    start_time = st.session_state.active_session_start or datetime.now()
    duration = int((datetime.now() - start_time).total_seconds())

    max_people = st.session_state.live_max_people
    max_phones = st.session_state.live_max_phones
    max_laptops = st.session_state.live_max_laptops
    phone_events = st.session_state.live_phone_events

    finish_session(
        session_id,
        duration,
        max_people,
        max_phones,
        max_laptops,
        phone_events
    )

    add_event(
        session_id,
        "SESSION_ENDED",
        f"Classroom session ended after {format_duration(duration)}"
    )

    st.session_state.last_session_summary = {
        "id": session_id,
        "duration": duration,
        "max_people": max_people,
        "max_phones": max_phones,
        "max_laptops": max_laptops,
        "phone_events": phone_events,
    }

    st.session_state.active_session_id = None
    st.session_state.active_session_start = None

    # Switch the camera checkbox off so a new session is not auto-created
    st.session_state.start_camera = False


# =========================================================
# TEACHER LOGIN
# =========================================================

def show_login():
    st.markdown("""
    <style>
        .login-box {
            max-width: 480px;
            margin: 70px auto 20px auto;
            padding: 35px;
            border-radius: 22px;
            background: white;
            border: 1px solid #e5e7eb;
            box-shadow: 0 12px 35px rgba(0,0,0,0.08);
            text-align: center;
        }
        .login-title {
            font-size: 34px;
            font-weight: 800;
            margin-bottom: 5px;
            color: #111827;
        }
        .login-subtitle {
            color: #6b7280;
            margin-bottom: 10px;
        }
    </style>
    <div class="login-box">
        <div class="login-title">🎓 ClassSense AI</div>
        <div class="login-subtitle">Teacher Dashboard Login</div>
    </div>
    """, unsafe_allow_html=True)

    _, center, _ = st.columns([1, 2, 1])

    with center:
        username = st.text_input("👤 Username")
        password = st.text_input("🔒 Password", type="password")

        if st.button("🚀 Login", type="primary", use_container_width=True):
            if username == DEMO_USERNAME and password == DEMO_PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("❌ Invalid username or password")

        st.caption("Demo login: teacher / classsense123")


if not st.session_state.authenticated:
    show_login()
    st.stop()


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="hero">
<h1>🎓 ClassSense AI</h1>
<p>AI-Powered Classroom Activity & Behaviour Analytics</p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown("## 🎓 ClassSense AI")
st.sidebar.caption("Smart Classroom Analytics Platform")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "📊 Dashboard",
        "📹 Classroom Monitor",
        "👨‍🎓 Student Analytics",
        "📝 Attendance",
        "🗂️ Reports"
    ]
)

st.sidebar.divider()

if st.sidebar.button("🚪 Logout", use_container_width=True):
    st.session_state.authenticated = False
    st.rerun()

st.sidebar.success("🟢 System Online")
st.sidebar.caption("AI Engine: YOLO")
st.sidebar.caption("Database: SQLite")
st.sidebar.caption("Version: 2.1")


# =========================================================
# DASHBOARD
# =========================================================

if page == "📊 Dashboard":

    st.markdown(
        '<div class="section-title">📊 Classroom Dashboard</div>',
        unsafe_allow_html=True
    )

    st.caption("Live overview of saved classroom monitoring data.")

    sessions = get_sessions()
    events = get_events()
    students = get_students()

    total_phone_events = sum(row[7] for row in sessions) if sessions else 0

    kpi_cards([
        ("🎓 Sessions", len(sessions), "Saved classroom sessions"),
        ("👨‍🎓 Students", len(students), "Anonymous IDs"),
        ("⚠️ Events", len(events), "Logged events"),
        ("📱 Phone Events", total_phone_events, "Possible detections only"),
    ])

    st.divider()

    left, right = st.columns([2, 1])

    with left:

        st.subheader("📈 Session Overview")

        if sessions:

            session_data = []

            for row in sessions[:10]:
                session_data.append({
                    "Session": f"Session {row[0]}",
                    "Max People": row[4],
                    "Max Phones": row[5],
                    "Max Laptops": row[6],
                    "Phone Events": row[7]
                })

            chart_df = pd.DataFrame(session_data)

            st.bar_chart(chart_df.set_index("Session"))

        else:
            st.info("No classroom sessions recorded yet.")

    with right:

        st.subheader("🟢 System Status")

        st.markdown(
            """
            <div class="card">

            <div class="status">● System Online</div>

            <br>

            <b>Computer Vision</b><br>
            YOLO Object Detection

            <br><br>

            <b>Database</b><br>
            SQLite

            <br><br>

            <b>Student IDs</b><br>
            Anonymous

            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    st.subheader("⚡ Recent Events")

    if events:

        recent_events = []

        for row in events[:10]:
            recent_events.append({
                "Time": format_ts(row[2]),
                "Type": EVENT_TYPE_LABELS.get(row[3], row[3]),
                "Description": row[4]
            })

        show_table(pd.DataFrame(recent_events))

    else:
        st.info("No events recorded yet.")

    st.caption(TEACHER_NOTE)


# =========================================================
# CLASSROOM MONITOR
# =========================================================

elif page == "📹 Classroom Monitor":

    st.markdown(
        '<div class="section-title">📹 Classroom Monitor</div>',
        unsafe_allow_html=True
    )

    st.caption("Computer vision powered classroom monitoring.")

    st.info(
        "Object detection shows possible events only. Detecting a phone "
        "does not prove that a student is using it."
    )

    # -----------------------------------------------------
    # CLASSROOM SESSION
    # -----------------------------------------------------

    st.subheader("🎓 Classroom Session")

    active_id = st.session_state.active_session_id

    button_col1, button_col2 = st.columns(2)

    with button_col1:
        st.button(
            "▶️ Start Classroom Session",
            type="primary",
            use_container_width=True,
            disabled=active_id is not None,
            on_click=start_new_session,
            key="start_session_btn"
        )

    with button_col2:
        st.button(
            "🛑 End Classroom Session",
            use_container_width=True,
            disabled=active_id is None,
            on_click=end_current_session,
            key="end_session_btn"
        )

    if active_id is not None:

        started = st.session_state.active_session_start

        kpi_cards([
            ("Session Status", f"🟢 Active #{active_id}", "Recording"),
            (
                "Started",
                started.strftime("%H:%M:%S") if started else "—",
                started.strftime("%d %b %Y") if started else ""
            ),
            (
                "Peak People",
                st.session_state.live_max_people,
                "Highest count so far"
            ),
            (
                "Phone Events",
                st.session_state.live_phone_events,
                "Possible detections"
            ),
        ])

    else:

        summary = st.session_state.last_session_summary

        if summary:

            st.success(
                f"✅ Session #{summary['id']} saved to the database."
            )

            kpi_cards([
                ("Duration", format_duration(summary["duration"]), "Total time"),
                ("Peak People", summary["max_people"], "Highest count"),
                ("Peak Phones", summary["max_phones"], "Possible detections"),
                ("Phone Events", summary["phone_events"], "Possible detections"),
            ])

        else:
            st.info(
                "No active session. Press Start Classroom Session to begin "
                "recording."
            )

    st.divider()

    mode = st.radio(
        "Select Mode",
        [
            "🖼️ Upload Image",
            "📷 Live Camera"
        ],
        horizontal=True
    )

    # -----------------------------------------------------
    # IMAGE MODE
    # -----------------------------------------------------

    if mode == "🖼️ Upload Image":

        st.subheader("🖼️ Classroom Image Analysis")

        st.caption(
            "Upload a classroom photo. YOLO will count people, phones and "
            "laptops it can see."
        )

        uploaded_file = st.file_uploader(
            "Upload a classroom image",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file:

            pil_image = Image.open(
                io.BytesIO(uploaded_file.getvalue())
            ).convert("RGB")

            image_left, image_right = st.columns(2)

            with image_left:
                st.image(
                    pil_image,
                    caption="Input Image",
                    use_container_width=True
                )

            if st.button("🔍 Analyse Image", type="primary"):

                try:

                    model = load_yolo_model()

                    with st.spinner("Running YOLO detection..."):
                        results = model(pil_image, conf=0.50, verbose=False)

                    result = results[0]

                    detected_image = result.plot()

                    with image_right:
                        st.image(
                            detected_image,
                            caption="YOLO Detection Result",
                            channels="BGR",
                            use_container_width=True
                        )

                    counts = {}
                    confidences = {}

                    for box in result.boxes:
                        class_name = model.names[int(box.cls[0])]
                        counts[class_name] = counts.get(class_name, 0) + 1
                        confidences.setdefault(class_name, []).append(
                            float(box.conf[0])
                        )

                    st.subheader("🔎 Detection Summary")

                    if counts:

                        kpi_cards([
                            ("👥 People", counts.get("person", 0), "Detected"),
                            ("📱 Phones", counts.get("cell phone", 0), "Possible only"),
                            ("💻 Laptops", counts.get("laptop", 0), "Detected"),
                            ("🔢 Total Objects", sum(counts.values()), "All classes"),
                        ])

                        detail_rows = []

                        for name, count in sorted(
                            counts.items(),
                            key=lambda item: item[1],
                            reverse=True
                        ):
                            avg_conf = sum(confidences[name]) / len(confidences[name])
                            detail_rows.append({
                                "Object": name.title(),
                                "Count": count,
                                "Avg Confidence": f"{avg_conf * 100:.0f}%"
                            })

                        show_table(pd.DataFrame(detail_rows))

                        description = "Detected objects: " + ", ".join(
                            f"{name}={count}" for name, count in counts.items()
                        )

                        current_session = st.session_state.active_session_id

                        add_event(
                            current_session,
                            "IMAGE_ANALYSIS",
                            description
                        )

                        if current_session is not None:
                            st.session_state.live_max_people = max(
                                st.session_state.live_max_people,
                                counts.get("person", 0)
                            )
                            st.session_state.live_max_phones = max(
                                st.session_state.live_max_phones,
                                counts.get("cell phone", 0)
                            )
                            st.session_state.live_max_laptops = max(
                                st.session_state.live_max_laptops,
                                counts.get("laptop", 0)
                            )

                        st.caption(
                            "✅ Analysis saved to Events"
                            + (
                                f" (Session #{current_session})."
                                if current_session is not None
                                else " (no active session)."
                            )
                        )

                    else:
                        st.warning("No objects detected.")

                except Exception as error:
                    st.error("YOLO could not run.")
                    st.code(str(error))

    # -----------------------------------------------------
    # LIVE CAMERA MODE
    # -----------------------------------------------------

    elif mode == "📷 Live Camera":

        st.subheader("📷 Live AI Classroom Camera")

        st.warning(
            "Object detection indicates possible events; "
            "it does not by itself prove phone usage or behaviour."
        )

        start_camera = st.checkbox("▶ Start Camera", key="start_camera")

        if start_camera:

            # If no session is active, start one automatically and refresh
            if st.session_state.active_session_id is None:
                start_new_session(
                    "Session started automatically by live camera"
                )
                st.rerun()

            try:

                from ultralytics import YOLO

                with st.spinner("Loading YOLO tracking model..."):
                    model = YOLO("yolo11n.pt")

                camera = cv2.VideoCapture(0)

                if not camera.isOpened():

                    camera.release()
                    st.error("❌ Could not open laptop camera.")

                else:

                    st.success("🟢 Camera connected.")

                    session_id = st.session_state.active_session_id

                    session_start = (
                        st.session_state.active_session_start
                        or datetime.now()
                    )

                    max_people = st.session_state.live_max_people
                    max_phones = st.session_state.live_max_phones
                    max_laptops = st.session_state.live_max_laptops
                    phone_event_count = st.session_state.live_phone_events

                    last_phone_count = 0
                    live_events = []

                    video_col, stats_col = st.columns([2, 1])

                    with video_col:
                        frame_placeholder = st.empty()

                    with stats_col:
                        st.markdown("**Live counts**")
                        person_placeholder = st.empty()
                        phone_placeholder = st.empty()
                        laptop_placeholder = st.empty()

                    st.divider()

                    st.subheader("⚠️ Live Events")
                    event_placeholder = st.empty()

                    st.divider()

                    st.subheader("📊 Session Statistics")

                    stat1, stat2, stat3, stat4 = st.columns(4)

                    duration_placeholder = stat1.empty()
                    max_people_placeholder = stat2.empty()
                    phone_events_placeholder = stat3.empty()
                    max_laptops_placeholder = stat4.empty()

                    try:

                        while True:

                            success, frame = camera.read()

                            if not success:
                                st.error("Could not read camera frame.")
                                break

                            results = model.track(
                                frame,
                                persist=True,
                                tracker="bytetrack.yaml",
                                conf=0.50,
                                verbose=False
                            )

                            result = results[0]

                            person_ids = set()
                            phone_ids = set()
                            laptop_ids = set()

                            clean_frame = frame.copy()

                            for box in result.boxes:

                                class_id = int(box.cls[0])
                                class_name = model.names[class_id]

                                if class_name not in (
                                    "person",
                                    "cell phone",
                                    "laptop"
                                ):
                                    continue

                                track_id = None

                                if box.id is not None:
                                    track_id = int(box.id[0])

                                if track_id is not None:
                                    if class_name == "person":
                                        person_ids.add(track_id)
                                    elif class_name == "cell phone":
                                        phone_ids.add(track_id)
                                    elif class_name == "laptop":
                                        laptop_ids.add(track_id)

                                coordinates = box.xyxy[0]

                                x1 = int(coordinates[0])
                                y1 = int(coordinates[1])
                                x2 = int(coordinates[2])
                                y2 = int(coordinates[3])

                                confidence = float(box.conf[0])

                                label = f"{class_name} {confidence:.2f}"

                                if track_id is not None:
                                    label += f" ID:{track_id}"

                                cv2.rectangle(
                                    clean_frame,
                                    (x1, y1),
                                    (x2, y2),
                                    (0, 255, 0),
                                    2
                                )

                                cv2.putText(
                                    clean_frame,
                                    label,
                                    (x1, max(y1 - 10, 20)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.55,
                                    (0, 255, 0),
                                    2
                                )

                            person_count = len(person_ids)
                            phone_count = len(phone_ids)
                            laptop_count = len(laptop_ids)

                            max_people = max(max_people, person_count)
                            max_phones = max(max_phones, phone_count)
                            max_laptops = max(max_laptops, laptop_count)

                            time_text = datetime.now().strftime("%H:%M:%S")

                            if phone_count > 0:

                                if phone_count != last_phone_count:

                                    phone_event_count += 1

                                    event_text = (
                                        f"Possible phone detected "
                                        f"({phone_count})"
                                    )

                                    live_events.append(
                                        f"{time_text}  📱 {event_text}"
                                    )

                                    add_event(
                                        session_id,
                                        "PHONE_DETECTED",
                                        event_text
                                    )

                                    last_phone_count = phone_count

                            else:
                                last_phone_count = 0

                            if len(live_events) > 8:
                                live_events = live_events[-8:]

                            # Keep live statistics so End Session can save them
                            st.session_state.live_max_people = max_people
                            st.session_state.live_max_phones = max_phones
                            st.session_state.live_max_laptops = max_laptops
                            st.session_state.live_phone_events = (
                                phone_event_count
                            )

                            total_seconds = int(
                                (datetime.now() - session_start).total_seconds()
                            )

                            frame_placeholder.image(
                                clean_frame,
                                channels="BGR",
                                use_container_width=True
                            )

                            person_placeholder.metric("👥 People", person_count)
                            phone_placeholder.metric("📱 Phones", phone_count)
                            laptop_placeholder.metric("💻 Laptops", laptop_count)

                            if live_events:
                                event_placeholder.markdown(
                                    "\n\n".join(live_events)
                                )
                            else:
                                event_placeholder.info(
                                    "No events detected yet."
                                )

                            duration_placeholder.metric(
                                "⏱️ Duration",
                                format_duration(total_seconds)
                            )

                            max_people_placeholder.metric(
                                "👥 Max People",
                                max_people
                            )

                            phone_events_placeholder.metric(
                                "📱 Phone Events",
                                phone_event_count
                            )

                            max_laptops_placeholder.metric(
                                "💻 Max Laptops",
                                max_laptops
                            )

                    finally:
                        camera.release()

            except Exception as error:

                st.error("Live camera could not run.")
                st.code(str(error))


# =========================================================
# STUDENT ANALYTICS
# =========================================================

elif page == "👨‍🎓 Student Analytics":

    st.markdown(
        '<div class="section-title">👨‍🎓 Student Analytics</div>',
        unsafe_allow_html=True
    )

    st.caption("Student records are stored using anonymous IDs.")

    st.info(
        "🧪 **Synthetic demo data.** Student records below are randomly "
        "generated for this prototype and are not real students. "
        + TEACHER_NOTE
    )

    existing_students = get_students()

    # Create demo records only when the database has no students yet
    if not existing_students:

        for i in range(1, 11):

            attendance = random.randint(70, 100)
            phone_events = random.randint(0, 5)
            interaction_events = random.randint(0, 6)
            activity_level = random.randint(50, 100)

            category, error_text = safe_predict(
                attendance,
                phone_events,
                interaction_events,
                activity_level
            )

            if error_text:
                st.error(
                    "The Random Forest model could not run. "
                    "Run `python train_model.py` first."
                )
                st.code(error_text)
                st.stop()

            save_student(
                f"STU-{i:03d}",
                attendance,
                phone_events,
                interaction_events,
                activity_level,
                category
            )

        existing_students = get_students()

    df_raw = build_student_df(existing_students)

    needs_attention = int((df_raw["AI Category"] == "Needs Attention").sum())

    kpi_cards([
        ("Students", len(df_raw), "Synthetic demo records"),
        ("Avg Attendance", f"{df_raw['Attendance (%)'].mean():.0f}%", "Synthetic"),
        ("Avg Activity", f"{df_raw['Activity (%)'].mean():.0f}%", "Synthetic"),
        ("Avg Phone Events", f"{df_raw['Phone Events'].mean():.1f}", "Possible detections"),
        ("Needs Attention", needs_attention, "For teacher review"),
    ])

    st.divider()

    st.subheader("📋 Student Overview")

    filter_col1, filter_col2 = st.columns([2, 1])

    options = category_options(df_raw)

    with filter_col1:
        chosen_categories = st.multiselect(
            "Filter by AI category",
            options,
            default=options
        )

    with filter_col2:
        search_text = st.text_input(
            "Search Student ID",
            placeholder="e.g. STU-003"
        )

    filtered_df = df_raw[df_raw["AI Category"].isin(chosen_categories)]

    if search_text.strip():
        filtered_df = filtered_df[
            filtered_df["Student ID"].str.contains(
                search_text.strip(),
                case=False,
                na=False,
                regex=False
            )
        ]

    if filtered_df.empty:
        st.info("No students match the current filters.")
    else:
        show_table(
            student_display(filtered_df),
            STUDENT_COLUMN_CONFIG
        )

    st.caption(
        f"Showing {len(filtered_df)} of {len(df_raw)} students."
    )

    st.divider()

    st.subheader("📈 Behaviour Distribution")

    chart_col, avg_col = st.columns(2)

    with chart_col:
        category_counts = (
            df_raw["AI Category"]
            .value_counts()
            .reindex(CATEGORY_ORDER, fill_value=0)
        )
        st.bar_chart(category_counts)

    with avg_col:
        st.markdown("**Average values per category**")
        averages = (
            df_raw.groupby("AI Category")[
                [
                    "Attendance (%)",
                    "Phone Events",
                    "Interactions",
                    "Activity (%)"
                ]
            ]
            .mean()
            .round(1)
            .reset_index()
        )
        show_table(averages)

    st.divider()

    st.subheader("🔎 Student Profile")

    selected_student = st.selectbox(
        "Select anonymous Student ID",
        list(df_raw["Student ID"])
    )

    student_row = df_raw[df_raw["Student ID"] == selected_student].iloc[0]

    kpi_cards([
        (
            "AI Category",
            CATEGORY_LABELS.get(
                student_row["AI Category"],
                student_row["AI Category"]
            ),
            "Support indicator"
        ),
        ("Attendance", f"{student_row['Attendance (%)']:.0f}%", "Synthetic"),
        ("Phone Events", student_row["Phone Events"], "Possible detections"),
        ("Interactions", student_row["Interactions"], "Synthetic"),
        ("Activity", f"{student_row['Activity (%)']:.0f}%", "Synthetic"),
    ])

    st.caption(
        "This category is a Random Forest prediction trained on synthetic "
        "data. It should only be used as a conversation starter for the "
        "teacher."
    )

    with st.expander("🧪 Try the Random Forest model (simulation — not saved)"):

        sim1, sim2, sim3, sim4 = st.columns(4)

        sim_attendance = sim1.slider("Attendance (%)", 0, 100, 85)
        sim_phone = sim2.slider("Phone events", 0, 10, 1)
        sim_interactions = sim3.slider("Interactions", 0, 10, 3)
        sim_activity = sim4.slider("Activity level (%)", 0, 100, 75)

        if st.button("🔮 Predict category", key="sim_predict"):

            predicted, predict_error = safe_predict(
                sim_attendance,
                sim_phone,
                sim_interactions,
                sim_activity
            )

            if predict_error:
                st.error("The model could not run.")
                st.code(predict_error)
            else:
                st.success(
                    f"Predicted category: "
                    f"{CATEGORY_LABELS.get(predicted, predicted)}"
                )
                st.caption(
                    "Simulation only. Nothing is saved to the database."
                )


# =========================================================
# ATTENDANCE
# =========================================================

elif page == "📝 Attendance":

    st.markdown(
        '<div class="section-title">📝 Attendance Register</div>',
        unsafe_allow_html=True
    )

    st.caption("Teacher-managed attendance using anonymous student IDs.")

    st.caption(
        "Note: the attendance percentage on the Student Analytics page is "
        "synthetic demo data. This page stores real teacher-marked records."
    )

    students = get_students()

    if not students:

        st.info("No student records available. Open Student Analytics first.")

    else:

        sessions = get_sessions()

        if sessions:

            session_options = {
                f"Session {row[0]} — {format_ts(row[1])}": row[0]
                for row in sessions
            }

            selected_session_label = st.selectbox(
                "Select classroom session",
                list(session_options.keys())
            )

            selected_session_id = session_options[selected_session_label]

        else:

            st.warning(
                "No classroom session exists yet. Start one from the "
                "Classroom Monitor page."
            )

            selected_session_id = None

        if selected_session_id is not None:

            cards_area = st.container()

            st.subheader("Mark Attendance")

            saved_status = {}

            for record in get_attendance():
                if record[1] == selected_session_id:
                    saved_status[record[2]] = record[3]

            with st.form(f"attendance_form_{selected_session_id}"):

                form_columns = st.columns(2)

                attendance_records = []

                for index, row in enumerate(students):

                    student_id = row[0]
                    default_status = saved_status.get(student_id, "Present")

                    with form_columns[index % 2]:
                        status = st.radio(
                            student_id,
                            ["Present", "Absent"],
                            horizontal=True,
                            index=0 if default_status == "Present" else 1,
                            key=f"attendance_{selected_session_id}_{student_id}"
                        )

                    attendance_records.append((student_id, status))

                submitted = st.form_submit_button(
                    "💾 Save Attendance",
                    type="primary",
                    use_container_width=True
                )

            if submitted:
                save_attendance(selected_session_id, attendance_records)
                st.success("Attendance saved successfully.")

            # Cards for the selected session (calculated after saving)
            current_status = {}

            for record in get_attendance():
                if record[1] == selected_session_id:
                    current_status[record[2]] = record[3]

            present_count = sum(
                1 for value in current_status.values() if value == "Present"
            )
            absent_count = sum(
                1 for value in current_status.values() if value == "Absent"
            )

            with cards_area:
                kpi_cards([
                    ("Students", len(students), "On the register"),
                    ("Marked", len(current_status), "Saved for this session"),
                    ("Present", present_count, "This session"),
                    ("Absent", absent_count, "This session"),
                ])

    st.divider()

    st.subheader("📊 Attendance Summary")

    summary = get_student_attendance_summary()

    if summary:

        summary_df = build_attendance_summary_df(summary)

        show_table(summary_df, ATTENDANCE_SUMMARY_CONFIG)

        st.caption(
            "'Below 75%' is only a prompt for the teacher to check in. "
            "It is not a decision or penalty."
        )

        csv_button(
            "⬇️ Download Attendance CSV",
            summary_df,
            "classsense_attendance.csv",
            "download_attendance_page"
        )

    else:
        st.info("No attendance has been saved yet.")


# =========================================================
# REPORTS
# =========================================================

elif page == "🗂️ Reports":

    st.markdown(
        '<div class="section-title">🗂️ Reports</div>',
        unsafe_allow_html=True
    )

    st.caption("Saved classroom monitoring and attendance records.")

    st.caption(
        "Phone events are possible detections, not confirmed phone use. "
        + TEACHER_NOTE
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "📹 Sessions",
        "⚠️ Events",
        "👨‍🎓 Students",
        "📝 Attendance"
    ])

    # ---------------------------------------------------------
    # SESSIONS REPORT
    # ---------------------------------------------------------

    with tab1:

        sessions = get_sessions()

        if sessions:

            active_id = st.session_state.get("active_session_id")

            raw_rows = []
            view_rows = []

            for row in sessions:

                (
                    session_id,
                    start_time,
                    end_time,
                    duration,
                    max_people,
                    max_phones,
                    max_laptops,
                    phone_events
                ) = row

                if end_time:
                    status = "Completed"
                elif session_id == active_id:
                    status = "Active"
                else:
                    status = "Not closed"

                raw_rows.append({
                    "Session ID": session_id,
                    "Start Time": start_time or "",
                    "End Time": end_time or "",
                    "Duration (sec)": duration,
                    "Max People": max_people,
                    "Max Phones": max_phones,
                    "Max Laptops": max_laptops,
                    "Phone Events": phone_events,
                    "Status": status
                })

                view_rows.append({
                    "Session": f"#{session_id}",
                    "Date": format_ts(start_time, "%d %b %Y"),
                    "Start": format_ts(start_time, "%H:%M:%S"),
                    "End": format_ts(end_time, "%H:%M:%S"),
                    "Duration": format_duration(duration),
                    "Max People": max_people,
                    "Max Phones": max_phones,
                    "Max Laptops": max_laptops,
                    "Phone Events": phone_events,
                    "Status": STATUS_LABELS[status]
                })

            raw_df = pd.DataFrame(raw_rows)
            view_df = pd.DataFrame(view_rows)

            completed = [row for row in sessions if row[2]]

            total_seconds = sum((row[3] or 0) for row in sessions)

            average_seconds = (
                sum((row[3] or 0) for row in completed) / len(completed)
                if completed else 0
            )

            kpi_cards([
                ("Sessions", len(sessions), f"{len(completed)} completed"),
                ("Total Time", format_duration(total_seconds), "All sessions"),
                ("Avg Session", format_duration(average_seconds), "Completed only"),
                ("Peak People", max(row[4] for row in sessions), "Highest in one session"),
                ("Phone Events", sum(row[7] for row in sessions), "Possible detections"),
            ])

            st.subheader("📹 Session Records")

            show_table(view_df)

            csv_button(
                "⬇️ Download Sessions CSV",
                raw_df,
                "classsense_sessions.csv",
                "download_sessions"
            )

            with st.expander("🔎 View events for one session"):

                chosen_session = st.selectbox(
                    "Select session",
                    [row[0] for row in sessions],
                    format_func=lambda value: f"Session #{value}",
                    key="session_detail_select"
                )

                session_events = [
                    e for e in get_events() if e[1] == chosen_session
                ]

                if session_events:
                    show_table(pd.DataFrame([
                        {
                            "Time": format_ts(e[2], "%H:%M:%S"),
                            "Type": EVENT_TYPE_LABELS.get(e[3], e[3]),
                            "Description": e[4]
                        }
                        for e in session_events
                    ]))
                else:
                    st.info("No events were logged for this session.")

        else:
            st.info("No sessions saved yet.")

    # ---------------------------------------------------------
    # EVENTS REPORT
    # ---------------------------------------------------------

    with tab2:

        events = get_events()

        if events:

            raw_rows = []

            for row in events:
                raw_rows.append({
                    "Event ID": row[0],
                    "Session ID": row[1] if row[1] is not None else "",
                    "Date": format_ts(row[2], "%d %b %Y"),
                    "Time": format_ts(row[2], "%H:%M:%S"),
                    "Event Type": row[3],
                    "Description": row[4]
                })

            events_df = pd.DataFrame(raw_rows)

            phone_total = int((events_df["Event Type"] == "PHONE_DETECTED").sum())
            image_total = int((events_df["Event Type"] == "IMAGE_ANALYSIS").sum())
            sessions_with_events = len(
                set(row[1] for row in events if row[1] is not None)
            )

            kpi_cards([
                ("Total Events", len(events), "All logged events"),
                ("Phone Detections", phone_total, "Possible only"),
                ("Image Analyses", image_total, "Uploaded images"),
                ("Sessions With Events", sessions_with_events, "Distinct sessions"),
            ])

            st.subheader("⚠️ Event Log")

            event_filter1, event_filter2 = st.columns(2)

            type_options = sorted(events_df["Event Type"].unique())

            with event_filter1:
                chosen_types = st.multiselect(
                    "Filter by event type",
                    type_options,
                    default=type_options,
                    format_func=lambda value: EVENT_TYPE_LABELS.get(value, value),
                    key="events_type_filter"
                )

            session_ids = sorted(
                set(row[1] for row in events if row[1] is not None),
                reverse=True
            )

            session_choices = (
                ["All sessions", "No session (standalone)"]
                + [f"Session {sid}" for sid in session_ids]
            )

            with event_filter2:
                chosen_session_filter = st.selectbox(
                    "Filter by session",
                    session_choices,
                    key="events_session_filter"
                )

            filtered_events = events_df[
                events_df["Event Type"].isin(chosen_types)
            ]

            if chosen_session_filter == "No session (standalone)":
                filtered_events = filtered_events[
                    filtered_events["Session ID"] == ""
                ]
            elif chosen_session_filter != "All sessions":
                wanted_id = int(chosen_session_filter.replace("Session ", ""))
                filtered_events = filtered_events[
                    filtered_events["Session ID"] == wanted_id
                ]

            if filtered_events.empty:
                st.info("No events match the current filters.")
            else:
                view_events = filtered_events.copy()
                view_events["Event Type"] = view_events["Event Type"].map(
                    lambda value: EVENT_TYPE_LABELS.get(value, value)
                )
                view_events["Session ID"] = view_events["Session ID"].map(
                    lambda value: f"#{value}" if value != "" else "—"
                )
                view_events = view_events.rename(
                    columns={"Event Type": "Type", "Session ID": "Session"}
                )
                show_table(view_events)

            st.caption(
                f"Showing {len(filtered_events)} of {len(events_df)} events."
            )

            csv_button(
                "⬇️ Download Events CSV (current filter)",
                filtered_events,
                "classsense_events.csv",
                "download_events"
            )

            with st.expander("📈 Event type breakdown"):
                breakdown = events_df["Event Type"].map(
                    lambda value: EVENT_TYPE_LABELS.get(value, value)
                ).value_counts()
                st.bar_chart(breakdown)

        else:
            st.info("No events saved yet.")

    # ---------------------------------------------------------
    # STUDENTS REPORT
    # ---------------------------------------------------------

    with tab3:

        students = get_students()

        if students:

            st.info(
                "🧪 Synthetic demo data with anonymous student IDs. "
                "Categories are support indicators for teacher review only."
            )

            students_df = build_student_df(students)

            kpi_cards([
                ("Students", len(students_df), "Anonymous IDs"),
                ("Avg Attendance", f"{students_df['Attendance (%)'].mean():.0f}%", "Synthetic"),
                ("Avg Activity", f"{students_df['Activity (%)'].mean():.0f}%", "Synthetic"),
                (
                    "Needs Attention",
                    int((students_df["AI Category"] == "Needs Attention").sum()),
                    "For teacher review"
                ),
            ])

            category_cols = st.columns(3)

            for column, category_name in zip(category_cols, CATEGORY_ORDER):
                count = int((students_df["AI Category"] == category_name).sum())
                column.metric(CATEGORY_LABELS[category_name], count)

            st.subheader("👨‍🎓 Student Records")

            report_options = category_options(students_df)

            report_categories = st.multiselect(
                "Filter by AI category",
                report_options,
                default=report_options,
                key="students_report_filter"
            )

            report_df = students_df[
                students_df["AI Category"].isin(report_categories)
            ]

            if report_df.empty:
                st.info("No students match the current filter.")
            else:
                show_table(
                    student_display(report_df),
                    STUDENT_COLUMN_CONFIG
                )

            csv_button(
                "⬇️ Download Students CSV",
                report_df,
                "classsense_students.csv",
                "download_students"
            )

        else:
            st.info("No student records available.")

    # ---------------------------------------------------------
    # ATTENDANCE REPORT
    # ---------------------------------------------------------

    with tab4:

        attendance_records = get_attendance()

        if attendance_records:

            raw_rows = []

            for row in attendance_records:
                raw_rows.append({
                    "Record ID": row[0],
                    "Session ID": row[1],
                    "Student ID": row[2],
                    "Status": row[3],
                    "Date": format_ts(row[4], "%d %b %Y"),
                    "Time": format_ts(row[4], "%H:%M:%S"),
                })

            attendance_df = pd.DataFrame(raw_rows)

            present_total = int((attendance_df["Status"] == "Present").sum())
            absent_total = int((attendance_df["Status"] == "Absent").sum())
            present_rate = (
                present_total / len(attendance_df) * 100
                if len(attendance_df) else 0
            )

            kpi_cards([
                ("Records", len(attendance_df), "All saved marks"),
                ("Present", present_total, "Total marks"),
                ("Absent", absent_total, "Total marks"),
                ("Present Rate", f"{present_rate:.1f}%", "Across all records"),
                (
                    "Sessions Covered",
                    attendance_df["Session ID"].nunique(),
                    "With attendance"
                ),
            ])

            st.subheader("📝 Attendance Records")

            att_filter1, att_filter2 = st.columns(2)

            with att_filter1:
                session_filter_options = ["All sessions"] + [
                    f"Session {sid}"
                    for sid in sorted(
                        attendance_df["Session ID"].unique(),
                        reverse=True
                    )
                ]
                chosen_att_session = st.selectbox(
                    "Filter by session",
                    session_filter_options,
                    key="attendance_session_filter"
                )

            with att_filter2:
                chosen_att_status = st.selectbox(
                    "Filter by status",
                    ["All", "Present", "Absent"],
                    key="attendance_status_filter"
                )

            filtered_attendance = attendance_df

            if chosen_att_session != "All sessions":
                wanted_session = int(chosen_att_session.replace("Session ", ""))
                filtered_attendance = filtered_attendance[
                    filtered_attendance["Session ID"] == wanted_session
                ]

            if chosen_att_status != "All":
                filtered_attendance = filtered_attendance[
                    filtered_attendance["Status"] == chosen_att_status
                ]

            if filtered_attendance.empty:
                st.info("No attendance records match the current filters.")
            else:
                view_attendance = filtered_attendance.copy()
                view_attendance["Status"] = view_attendance["Status"].map(
                    lambda value: ATTENDANCE_LABELS.get(value, value)
                )
                view_attendance["Session ID"] = view_attendance["Session ID"].map(
                    lambda value: f"#{value}"
                )
                view_attendance = view_attendance.rename(
                    columns={"Session ID": "Session"}
                )
                show_table(view_attendance)

            st.caption(
                f"Showing {len(filtered_attendance)} of "
                f"{len(attendance_df)} records."
            )

            csv_button(
                "⬇️ Download Attendance CSV",
                filtered_attendance,
                "classsense_attendance.csv",
                "download_attendance_records"
            )

            st.subheader("📊 Student Attendance Summary")

            summary = get_student_attendance_summary()

            if summary:

                summary_df = build_attendance_summary_df(summary)

                show_table(summary_df, ATTENDANCE_SUMMARY_CONFIG)

                st.caption(
                    "'Below 75%' is only a prompt for the teacher to check "
                    "in. It is not a decision or penalty."
                )

                csv_button(
                    "⬇️ Download Attendance Summary CSV",
                    summary_df,
                    "classsense_attendance_summary.csv",
                    "download_attendance_summary"
                )

        else:
            st.info("No attendance records available yet.")


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div class="footer">
    ClassSense AI • Intelligent Classroom Analytics<br>
    Teacher-support prototype • Uses anonymous IDs and synthetic demo data<br>
    Not used for marks, promotion or disciplinary decisions
    </div>
    """,
    unsafe_allow_html=True
)