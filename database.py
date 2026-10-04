import sqlite3
from datetime import datetime


DATABASE_NAME = "classsense.db"


def get_connection():
    return sqlite3.connect(DATABASE_NAME)


def initialize_database():

    connection = get_connection()
    cursor = connection.cursor()

    # Classroom sessions
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            start_time TEXT NOT NULL,
            end_time TEXT,
            duration_seconds INTEGER DEFAULT 0,
            max_people INTEGER DEFAULT 0,
            max_phones INTEGER DEFAULT 0,
            max_laptops INTEGER DEFAULT 0,
            phone_events INTEGER DEFAULT 0
        )
    """)

    # Classroom events
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            event_time TEXT NOT NULL,
            event_type TEXT NOT NULL,
            description TEXT,
            FOREIGN KEY (session_id)
            REFERENCES sessions(id)
        )
    """)

    # Student analytics
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT UNIQUE NOT NULL,
            attendance REAL DEFAULT 0,
            phone_events INTEGER DEFAULT 0,
            interactions INTEGER DEFAULT 0,
            activity_level REAL DEFAULT 0,
            category TEXT,
            updated_at TEXT
        )
    """)

    connection.commit()
    connection.close()


def create_session():

    connection = get_connection()
    cursor = connection.cursor()

    start_time = datetime.now().isoformat()

    cursor.execute(
        """
        INSERT INTO sessions (start_time)
        VALUES (?)
        """,
        (start_time,)
    )

    session_id = cursor.lastrowid

    connection.commit()
    connection.close()

    return session_id


def add_event(
    session_id,
    event_type,
    description
):

    connection = get_connection()
    cursor = connection.cursor()

    event_time = datetime.now().isoformat()

    cursor.execute(
        """
        INSERT INTO events (
            session_id,
            event_time,
            event_type,
            description
        )
        VALUES (?, ?, ?, ?)
        """,
        (
            session_id,
            event_time,
            event_type,
            description
        )
    )

    connection.commit()
    connection.close()


def finish_session(
    session_id,
    duration_seconds,
    max_people,
    max_phones,
    max_laptops,
    phone_events
):

    connection = get_connection()
    cursor = connection.cursor()

    end_time = datetime.now().isoformat()

    cursor.execute(
        """
        UPDATE sessions

        SET
            end_time = ?,
            duration_seconds = ?,
            max_people = ?,
            max_phones = ?,
            max_laptops = ?,
            phone_events = ?

        WHERE id = ?
        """,
        (
            end_time,
            duration_seconds,
            max_people,
            max_phones,
            max_laptops,
            phone_events,
            session_id
        )
    )

    connection.commit()
    connection.close()


def save_student(
    student_id,
    attendance,
    phone_events,
    interactions,
    activity_level,
    category
):

    connection = get_connection()
    cursor = connection.cursor()

    updated_at = datetime.now().isoformat()

    cursor.execute(
        """
        INSERT INTO students (
            student_id,
            attendance,
            phone_events,
            interactions,
            activity_level,
            category,
            updated_at
        )

        VALUES (?, ?, ?, ?, ?, ?, ?)

        ON CONFLICT(student_id)
        DO UPDATE SET

            attendance = excluded.attendance,
            phone_events = excluded.phone_events,
            interactions = excluded.interactions,
            activity_level = excluded.activity_level,
            category = excluded.category,
            updated_at = excluded.updated_at
        """,
        (
            student_id,
            attendance,
            phone_events,
            interactions,
            activity_level,
            category,
            updated_at
        )
    )

    connection.commit()
    connection.close()


def get_students():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            student_id,
            attendance,
            phone_events,
            interactions,
            activity_level,
            category,
            updated_at

        FROM students

        ORDER BY student_id
        """
    )

    rows = cursor.fetchall()

    connection.close()

    return rows


def get_sessions():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            id,
            start_time,
            end_time,
            duration_seconds,
            max_people,
            max_phones,
            max_laptops,
            phone_events

        FROM sessions

        ORDER BY id DESC
        """
    )

    rows = cursor.fetchall()

    connection.close()

    return rows


def get_events():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            id,
            session_id,
            event_time,
            event_type,
            description

        FROM events

        ORDER BY id DESC
        """
    )

    rows = cursor.fetchall()

    connection.close()

    return rows


# Create database automatically
initialize_database()