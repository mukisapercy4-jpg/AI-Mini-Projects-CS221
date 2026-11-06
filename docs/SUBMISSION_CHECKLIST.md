# Pre-Submission Checklist (Deadline: 5th December 2026)

Before uploading to MUELE ([Section 230921](https://muele.mak.ac.ug/course/section.php?id=230921)), the **Integration & Submission Lead** and all team members must verify each item:

### 1. General Requirements
- [ ] Team details (Names, Student/Registration numbers, Emails, and Assigned roles) are placed clearly at the beginning of:
  - [ ] `notebooks/ai_coursework_all_assignments.ipynb`
  - [ ] `theoretical/theoretical_solutions.pdf`
- [ ] Every assignment is distinctly numbered and titled:
  - 1. Foundations (`foundations`)
  - 2. Sentiment Classification (`sentiment`)
  - 3. Route Planning (`route`)
  - 4. Controlling MountainCar (`mountaincar`)
  - 5. Pac-Man (`pacman`)
  - 6. Course Scheduling & Bayesian Networks (`scheduling` / `bayesian`)
  - 7. Car Tracking & Logic (`tracking` / `logic`)

### 2. Theoretical Component
- [ ] All 7 theoretical parts are merged into **one single PDF document**.
- [ ] Mathematical steps, proofs, and graphs are clearly legible.
- [ ] PDF exported as `theoretical/theoretical_solutions.pdf`.

### 3. Programming Component
- [ ] All programming tasks run within **one single notebook** (`ai_coursework_all_assignments.ipynb`).
- [ ] The entire notebook has been executed via `Restart Kernel and Run All Cells` without runtime errors.
- [ ] Detailed documentation, docstrings, and inline comments explain all core logic.
- [ ] Output figures, evaluation scores, and metrics are rendered and visible.

### 4. Packaging & Upload
- [ ] Run `python scripts/bundle_submission.py` to produce the final `.zip` file.
- [ ] Check that zip file size is within MUELE upload limits.
- [ ] Uploaded by the designated team member before **5th December 2026, 23:59 EAT**.
