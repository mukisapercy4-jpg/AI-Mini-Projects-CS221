"""
Submission Bundler Script
Packages the final project deliverables into a single zip archive for Muele submission.
Verifies all required artifacts are present before bundling.
"""

import os
import sys
import zipfile
from datetime import datetime

REQUIRED_FILES = [
    "notebooks/ai_coursework_all_assignments.ipynb",
]

RECOMMENDED_FILES = [
    "theoretical/theoretical_solutions.pdf",
    "theoretical/theoretical_part_master.md",
    "README.md",
]

def bundle():
    print("=" * 60)
    print("AI MINI-PROJECTS: MUELE SUBMISSION BUNDLER")
    print("=" * 60)

    # 1. Verification
    missing_required = [f for f in REQUIRED_FILES if not os.path.exists(f)]
    if missing_required:
        print("[ERROR] Missing required file(s):")
        for f in missing_required:
            print(f"  - {f}")
        print("Please ensure the master notebook exists before packaging.")
        sys.exit(1)

    output_dir = "submission"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = os.path.join(output_dir, f"AI_MiniProjects_Final_Submission_{timestamp}.zip")

    # Files to include
    include_paths = [
        "notebooks",
        "theoretical",
        "assignments",
        "README.md",
        "requirements.txt"
    ]

    print(f"\n[INFO] Compressing project files into {zip_filename}...")
    file_count = 0
    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        for path in include_paths:
            if os.path.isfile(path):
                zipf.write(path, arcname=path)
                file_count += 1
                print(f"  + Added file: {path}")
            elif os.path.isdir(path):
                for root, _, files in os.walk(path):
                    for file in files:
                        if ".ipynb_checkpoints" in root or "__pycache__" in root:
                            continue
                        full_path = os.path.join(root, file)
                        arcname = os.path.relpath(full_path, ".")
                        zipf.write(full_path, arcname=arcname)
                        file_count += 1
                        print(f"  + Added: {arcname}")

    print("\n" + "=" * 60)
    print(f"[SUCCESS] Packaged {file_count} files successfully!")
    print(f"Submission Archive: {os.path.abspath(zip_filename)}")
    print("Ready to upload to Muele: https://muele.mak.ac.ug/course/section.php?id=230921")
    print("=" * 60)

if __name__ == "__main__":
    bundle()
