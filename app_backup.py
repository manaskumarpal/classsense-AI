import streamlit as st
import pandas as pd
import random

from ml_model import predict_behavior


# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="ClassSense AI",
    page_icon="🎓",
    layout="wide"
)


# --------------------------------
# PAGE HEADER
# --------------------------------

st.title("🎓 ClassSense AI")

st.subheader(
    "AI-Powered Classroom Activity & Behaviour Analytics"
)

st.info(
    "Prototype demonstration of an AI classroom monitoring system."
)


# --------------------------------
# SIDEBAR
# --------------------------------

st.sidebar.title("Teacher Panel")

page = st.sidebar.radio(
    "Select Module",
    [
        "Dashboard",
        "Classroom Monitor",
        "Student Analytics"
    ]
)


# --------------------------------
# DASHBOARD
# --------------------------------

if page == "Dashboard":

    st.header("📊 Classroom Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Students Detected",
        "42"
    )

    col2.metric(
        "Currently Active",
        "35"
    )

    col3.metric(
        "Phone Alerts",
        "3"
    )

    col4.metric(
        "Attention Alerts",
        "4"
    )

    st.divider()

    st.subheader("Today's Classroom Activity")

    chart_data = pd.DataFrame(
        {
            "Activity": [
                "Active",
                "Phone Usage",
                "Interaction",
                "Inactive"
            ],
            "Students": [
                35,
                3,
                2,
                2
            ]
        }
    )

    st.bar_chart(
        chart_data.set_index("Activity")
    )

    st.success(
        "Classroom monitoring system is active."
    )


# --------------------------------
# CLASSROOM MONITOR
# --------------------------------

elif page == "Classroom Monitor":

    st.header("📹 Classroom Monitor")

    st.write(
        "This prototype demonstrates how a camera-based "
        "AI system could analyse a classroom."
    )

    st.warning(
        "Deep Learning detection module will run when "
        "the classroom image is analysed."
    )

    uploaded_file = st.file_uploader(
        "Upload Classroom Image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    if uploaded_file:

        st.image(
            uploaded_file,
            caption="Classroom Image",
            use_container_width=True
        )

        if st.button("🔍 Analyse Classroom"):

            st.write(
                "Loading Deep Learning model..."
            )

            try:

                from ultralytics import YOLO

                model = YOLO("yolo11n.pt")

                results = model(
                    uploaded_file
                )

                result = results[0]

                detected_image = result.plot()

                st.image(
                    detected_image,
                    caption="YOLO Detection Result",
                    channels="BGR",
                    use_container_width=True
                )

                detected_objects = []

                for box in result.boxes:

                    class_id = int(
                        box.cls[0]
                    )

                    class_name = model.names[
                        class_id
                    ]

                    detected_objects.append(
                        class_name
                    )

                st.subheader(
                    "🔎 Detected Objects"
                )

                if detected_objects:

                    counts = {}

                    for item in detected_objects:

                        counts[item] = (
                            counts.get(item, 0) + 1
                        )

                    for item, count in counts.items():

                        st.write(
                            f"**{item}** : {count}"
                        )

                else:

                    st.warning(
                        "No objects detected."
                    )

            except Exception as error:

                st.error(
                    "YOLO could not run."
                )

                st.code(
                    str(error)
                )


# --------------------------------
# STUDENT ANALYTICS
# --------------------------------

elif page == "Student Analytics":

    st.header(
        "👨‍🎓 Student Behaviour Analytics"
    )

    students = []

    for i in range(1, 11):

        attendance = random.randint(
            70,
            100
        )

        phone_events = random.randint(
            0,
            5
        )

        interaction_events = random.randint(
            0,
            6
        )

        activity_level = random.randint(
            50,
            100
        )

        category = predict_behavior(
            attendance,
            phone_events,
            interaction_events,
            activity_level
        )

        students.append(
            {
                "Student ID":
                    f"CU2026{i:03d}",

                "Attendance":
                    f"{attendance}%",

                "Phone Events":
                    phone_events,

                "Interactions":
                    interaction_events,

                "Activity":
                    f"{activity_level}%",

                "AI Category":
                    category
            }
        )

    df_students = pd.DataFrame(
        students
    )

    st.dataframe(
        df_students,
        use_container_width=True,
        hide_index=True
    )

    st.subheader(
        "📈 Behaviour Summary"
    )

    category_counts = (
        df_students["AI Category"]
        .value_counts()
    )

    st.bar_chart(
        category_counts
    )

    st.caption(
        "Prototype analytics for demonstration."
    )