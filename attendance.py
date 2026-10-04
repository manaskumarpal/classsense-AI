import sqlite3
from datetime import datetime

DATABASE_NAME = "classsense.db"


def get_connection():
    return sqlite3.connect(DATABASE_NAME)


def initialize_attendance_table():
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            student_id TEXT NOT NULL,
            status TEXT NOT NULL,
            attendance_time TEXT NOT NULL,
            UNIQUE(session_id, student_id),
            FOREIGN KEY(session_id) REFERENCES sessions(id)
        )
    """)

    connection.commit()
    connection.close()


def save_attendance(session_id, records):
    connection = get_connection()
    cursor = connection.cursor()

    for student_id, status in records:
        cursor.execute("""
            INSERT INTO attendance (
                session_id,
                student_id,
                status,
                attendance_time
            )
            VALUES (?, ?, ?, ?)
            ON CONFLICT(session_id, student_id)
            DO UPDATE SET
                status = excluded.status,
                attendance_time = excluded.attendance_time
        """, (
            session_id,
            student_id,
            status,
            datetime.now().isoformat()
        ))

    connection.commit()
    connection.close()


def get_attendance():
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            id,
            session_id,
            student_id,
            status,
            attendance_time
        FROM attendance
        ORDER BY id DESC
    """)

    rows = cursor.fetchall()
    connection.close()
    return rows


def get_student_attendance_summary():
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            student_id,
            SUM(CASE WHEN status = 'Present' THEN 1 ELSE 0 END),
            COUNT(*)
        FROM attendance
        GROUP BY student_id
        ORDER BY student_id
    """)

    rows = cursor.fetchall()
    connection.close()
    return rows


initialize_attendance_table()