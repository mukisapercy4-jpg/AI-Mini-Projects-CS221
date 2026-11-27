# Pre-Submission Checklist (Deadline: 5th December 2026)

Before uploading to MUELE ([Section 230921](https://muele.mak.ac.ug/course/section.php?id=230921)), the **Integration & Submission Lead** and all team members must verify each item:

### 1. General Requirements
- [x] Team details (Names, Student/Registration numbers, Emails, and Assigned roles) are placed clearly at the beginning of:
  - [x] `notebooks/ai_coursework_all_assignments.ipynb`
  - [x] `theoretical/theoretical_solutions.pdf`
- [x] Every assignment is distinctly numbered and titled:
  - 1. Foundations (`foundations`)
  - 2. Sentiment Classification (`sentiment`)
  - 3. Route Planning (`route`)
  - 4. Controlling MountainCar (`mountaincar`)
  - 5. Pac-Man (`pacman`)
  - 6. Course Scheduling & Bayesian Networks (`scheduling` / `bayesian`)
  - 7. Car Tracking & Logic (`tracking` / `logic`)

### 2. Theoretical Component
- [x] All 7 grouped theoretical sections are merged into **one single PDF document**.
- [x] Mathematical steps and proofs were reviewed in the rendered PDF. (No separate graphs were required by the supplied theory draft.)
- [x] PDF exported as `theoretical/theoretical_solutions.pdf`.

### 3. Programming Component
- [x] All seven grouped programming tasks are included in **one single notebook** (`ai_coursework_all_assignments.ipynb`).
- [x] All notebook code cells were run sequentially without runtime errors; outputs are saved in the notebook.
- [x] Core submission classes and functions have docstrings, with explanatory section notes and implementation comments.
- [x] Evaluation scores and metrics from the supplied experiments are rendered and visible in the notebook.

### 4. Packaging & Upload
- [x] Run `python scripts/bundle_submission.py` to produce the final `.zip` file.
- [ ] Confirm the archive is within the MUELE upload limit.
- [ ] Upload by the designated team member before **5th December 2026, 23:59 EAT**.
