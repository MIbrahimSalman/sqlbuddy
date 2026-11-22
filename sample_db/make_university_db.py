"""Generate sample_db/university.db - a small, realistic university database for the demo.

    python sample_db/make_university_db.py

Deterministic (fixed random seed), standard library only. Tables:
    teachers(teacher_id, name, department, title, hire_year)
    students(student_id, name, major, enrollment_year, city)
    courses(course_id, code, title, department, credits, teacher_id -> teachers)
    enrollments(enrollment_id, student_id -> students, course_id -> courses, semester)
    grades(grade_id, enrollment_id -> enrollments, score, letter)
"""
import random
import sqlite3
from pathlib import Path

OUT = Path(__file__).with_name("university.db")
rng = random.Random(2026)

FIRST = ["Ayesha", "Ali", "Fatima", "Hamza", "Zainab", "Omar", "Sara", "Bilal", "Maryam", "Usman",
         "Hira", "Ahmed", "Noor", "Daniyal", "Amna", "Saad", "Iqra", "Hassan", "Mahnoor", "Raza",
         "Emma", "Lucas", "Mia", "Noah", "Lina", "Yusuf", "Sofia", "Adam", "Laila", "Imran"]
LAST = ["Khan", "Malik", "Ahmed", "Hussain", "Qureshi", "Siddiqui", "Butt", "Raza", "Shah", "Iqbal",
        "Chaudhry", "Sheikh", "Mirza", "Baig", "Javed", "Smith", "Garcia", "Nguyen", "Rossi", "Weber"]
CITIES = ["Lahore", "Karachi", "Islamabad", "Rawalpindi", "Faisalabad", "Multan", "Peshawar"]
DEPTS = {
    "Computer Science": [("CS101", "Introduction to Programming", 4), ("CS201", "Data Structures", 4),
                         ("CS301", "Databases", 3), ("CS401", "Natural Language Processing", 3),
                         ("CS402", "Machine Learning", 3)],
    "Mathematics": [("MA101", "Calculus I", 3), ("MA201", "Linear Algebra", 3), ("MA301", "Probability and Statistics", 3)],
    "Physics": [("PH101", "Mechanics", 4), ("PH201", "Electromagnetism", 4)],
    "Business": [("BU101", "Principles of Management", 3), ("BU201", "Financial Accounting", 3)],
}
TITLES = ["Lecturer", "Assistant Professor", "Associate Professor", "Professor"]
SEMESTERS = ["Fall 2024", "Spring 2025", "Fall 2025"]


def letter(score: float) -> str:
    for cut, l in [(85, "A"), (75, "B"), (65, "C"), (50, "D")]:
        if score >= cut:
            return l
    return "F"


def name() -> str:
    return f"{rng.choice(FIRST)} {rng.choice(LAST)}"


def main():
    OUT.unlink(missing_ok=True)
    con = sqlite3.connect(OUT)
    con.executescript("""
    PRAGMA foreign_keys = ON;
    CREATE TABLE teachers (
      teacher_id INTEGER PRIMARY KEY, name TEXT NOT NULL, department TEXT NOT NULL,
      title TEXT, hire_year INTEGER);
    CREATE TABLE students (
      student_id INTEGER PRIMARY KEY, name TEXT NOT NULL, major TEXT, enrollment_year INTEGER, city TEXT);
    CREATE TABLE courses (
      course_id INTEGER PRIMARY KEY, code TEXT UNIQUE NOT NULL, title TEXT NOT NULL, department TEXT,
      credits INTEGER, teacher_id INTEGER REFERENCES teachers(teacher_id));
    CREATE TABLE enrollments (
      enrollment_id INTEGER PRIMARY KEY, student_id INTEGER NOT NULL REFERENCES students(student_id),
      course_id INTEGER NOT NULL REFERENCES courses(course_id), semester TEXT);
    CREATE TABLE grades (
      grade_id INTEGER PRIMARY KEY, enrollment_id INTEGER NOT NULL REFERENCES enrollments(enrollment_id),
      score REAL, letter TEXT);
    """)

    # 10 teachers, 2-3 per department
    teachers, tid = [], 1
    for dept in DEPTS:
        for _ in range(3 if dept == "Computer Science" else 2):
            teachers.append((tid, "Dr. " + name(), dept, rng.choice(TITLES), rng.randint(2005, 2023)))
            tid += 1
    con.executemany("INSERT INTO teachers VALUES (?,?,?,?,?)", teachers)

    # 12 courses, taught by a teacher of the same department
    courses, cid = [], 1
    for dept, items in DEPTS.items():
        dept_teachers = [t[0] for t in teachers if t[2] == dept]
        for code, title, credits in items:
            courses.append((cid, code, title, dept, credits, rng.choice(dept_teachers)))
            cid += 1
    con.executemany("INSERT INTO courses VALUES (?,?,?,?,?,?)", courses)

    # 25 students
    majors = list(DEPTS)
    students = [(i, name(), rng.choice(majors), rng.choice([2022, 2023, 2024, 2025]), rng.choice(CITIES))
                for i in range(1, 26)]
    con.executemany("INSERT INTO students VALUES (?,?,?,?,?)", students)

    # each student takes 2-4 distinct courses (mostly in their major) -> ~75 enrollments + grades
    enrollments, grades, eid = [], [], 1
    for sid, _n, major, year, _c in students:
        in_major = [c[0] for c in courses if c[3] == major]
        others = [c[0] for c in courses if c[3] != major]
        k = rng.randint(2, 4)
        chosen = rng.sample(in_major, min(len(in_major), k - 1)) + rng.sample(others, 1)
        for course_id in chosen:
            sem = rng.choice(SEMESTERS[max(0, year - 2024):] or SEMESTERS[-1:])
            enrollments.append((eid, sid, course_id, sem))
            if sem != "Fall 2025" or rng.random() < 0.5:  # some current-semester grades are missing
                score = round(min(100, max(30, rng.gauss(72, 12))), 1)
                grades.append((eid, eid, score, letter(score)))
            eid += 1
    con.executemany("INSERT INTO enrollments VALUES (?,?,?,?)", enrollments)
    con.executemany("INSERT INTO grades VALUES (?,?,?,?)", grades)
    con.commit()
    counts = {t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
              for t in ["teachers", "students", "courses", "enrollments", "grades"]}
    con.close()
    print(f"wrote {OUT}: {counts} (total {sum(counts.values())} rows)")


if __name__ == "__main__":
    main()
