# AI Mini-Projects Coursework (CS221 Aligned)

> **Course Portal:** [Makerere University MUELE Section 230921](https://muele.mak.ac.ug/course/section.php?id=230921)  
> **Course Syllabus & Reference Materials:** [Stanford CS221: Artificial Intelligence - Principles and Techniques](https://stanford-cs221.github.io/spring2026/#coursework)  
> **Submission Deadline:** 5th December 2026  

---

## 📌 Project Overview & Requirements

This repository contains the collaborative implementation of the Artificial Intelligence Mini-Project assignments. 

### Mandatory Submission Rules (Per Course Instructions)
1. **Theoretical Component:** All theoretical questions across all assignments must be compiled into **one single PDF document** (`theoretical_solutions.pdf`).
2. **Programming Component:** All programming implementations must be organized and executed inside **one (same) Python Jupyter Notebook** (`notebooks/ai_coursework_all_assignments.ipynb`).
3. **Numbering & Titles:** Every assignment in both the PDF and notebook must be clearly numbered and titled.
4. **Code Quality & Documentation:** Every function, algorithm, and experimental step must include clean docstrings and explanatory markdown comments for readability.
5. **Team Identification:** Complete team details (full names, registration/student numbers, emails, and assigned modules) must appear at the top of the notebook and PDF.
6. **Zipped Submission:** If auxiliary datasets or libraries are used, all artifacts will be bundled and submitted as a single `.zip` file on MUELE.

---

## 👥 Team Roster & Role Assignments

| # | Assignment Module | Assigned Team Member(s) | Status |
|---|-------------------|-------------------------|--------|
| 1 | **Foundations** (`foundations`) | Henry | In Progress |
| 2 | **Sentiment Classification** (`sentiment`) | *[To be confirmed]* | Assigned |
| 3 | **Route Planning** (`route`) | Precious Atwiine | In Progress |
| 4 | **Controlling MountainCar** (`mountaincar`) | Marcus & *[Co-lead]* | In Progress |
| 5 | **Pac-Man** (`pacman`) | Jill & Mukisa Percy | In Progress |
| 6 | **Course Scheduling & Bayesian Networks** (`scheduling` / `bayesian`) | Joseph Mary | In Progress |
| 7 | **Car Tracking & Logic** (`tracking` / `logic`) | Josh Kwizera | In Progress |
| 8 | **Integration & Submission Lead** | *[To be confirmed]* | Assigned |

> **Integration & Submission Lead Role:** Responsible for merging code contributions into the master notebook, ensuring numbering/titling consistency, validating end-to-end execution, compiling the final theoretical PDF, and packaging the submission.

---

## 📂 Repository Layout

```text
Artificial_Intelligence_Project_Work/
├── .gitignore
├── README.md
├── requirements.txt
├── notebooks/
│   └── ai_coursework_all_assignments.ipynb   # Master unified submission notebook
├── theoretical/
│   ├── theoretical_part_master.md             # Master theoretical draft
│   └── theoretical_part_master.tex            # LaTeX template for single PDF export
├── assignments/                              # Isolated staging folders for team members
│   ├── 01_foundations/
│   ├── 02_sentiment/
│   ├── 03_route_planning/
│   ├── 04_mountaincar/
│   ├── 05_pacman/
│   ├── 06_course_scheduling/
│   └── 07_car_tracking/
├── scripts/
│   └── bundle_submission.py                  # Script to package submission zip
└── docs/
    └── SUBMISSION_CHECKLIST.md               # Final pre-submission verification
```

---

## 🚀 Getting Started

### 1. Clone & Setup Virtual Environment
```bash
git clone <YOUR_GITHUB_REPO_URL>
cd Artificial_Intelligence_Project_Work

# Create and activate virtual environment
python -m venv .venv
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On macOS / Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Team Member Workflow
To avoid merge conflicts on the master notebook:
1. Work inside your assigned staging folder under `assignments/<module_name>/`.
2. Write unit tests, helper functions, and writeups in your folder.
3. Coordinate with the **Integration Lead** to port validated solutions into `notebooks/ai_coursework_all_assignments.ipynb` and `theoretical/theoretical_part_master.tex` / `.md`.

### 3. Generate Submission Package
```bash
python scripts/bundle_submission.py
```
This produces `submission/AI_MiniProjects_GroupSubmission.zip` ready for upload to MUELE.
