"""
Flask frontend for the CV screening pipeline in prototype_cvscreening.py.

This file does not change anything about the pipeline itself. It only:
  - serves the upload page
  - saves uploaded .txt files to a temporary folder
  - calls run_batch() with the paths run_batch() already expects
  - shapes the returned data (plus the saved contact-info file) into JSON
    for the page to render
"""

import contextlib
import io
import json
import os
import shutil
import tempfile

from flask import Flask, jsonify, render_template, request

import prototype_cvscreening as cvscreen

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB is generous for .txt uploads

ALLOWED_EXTENSION = ".txt"


def _is_txt(filename):
    return bool(filename) and filename.lower().endswith(ALLOWED_EXTENSION)


def _extract_notes(log_text, original_names):
    """
    run_batch() explains skipped/rejected files with print() statements, not
    a return value (e.g. "Rejected cv_3.txt (applicant 03): ..."). We capture
    stdout while it runs and pull out just those lines, so the site can show
    *why* a candidate didn't make the final ranking instead of just going
    quiet about it. The pipeline reports these using the temp file path we
    handed it, so we swap each one back out for the name that was actually
    uploaded before showing it.
    """
    notes = []
    for line in log_text.splitlines():
        line = line.strip()
        if line.startswith(("Skipping", "Rejected", "No ")):
            for path, original in original_names.items():
                line = line.replace(path, original)
            notes.append(line)
    return notes


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/screen", methods=["POST"])
def screen():
    job_file = request.files.get("job_description")
    cv_files = [f for f in request.files.getlist("cvs") if f and f.filename]

    if not job_file or not job_file.filename:
        return jsonify(success=False, error="Please attach a job description file."), 400
    if not _is_txt(job_file.filename):
        return jsonify(success=False, error="The job description needs to be a .txt file."), 400
    if not cv_files:
        return jsonify(success=False, error="Please attach at least one candidate CV."), 400

    bad_names = [f.filename for f in cv_files if not _is_txt(f.filename)]
    if bad_names:
        return jsonify(
            success=False,
            error="These files aren't .txt files: " + ", ".join(bad_names),
        ), 400

    work_dir = tempfile.mkdtemp(prefix="cv_screening_")
    try:
        job_path = os.path.join(work_dir, "job_description.txt")
        job_file.save(job_path)

        cv_paths = []
        original_names = {}  # temp path -> the filename actually uploaded
        for i, f in enumerate(cv_files):
            # basename() strips any path component a browser might send, and the
            # index prefix keeps two candidates with the same filename apart.
            original_name = os.path.basename(f.filename)
            safe_name = f"{i:03d}_{original_name}"
            path = os.path.join(work_dir, safe_name)
            f.save(path)
            cv_paths.append(path)
            original_names[path] = original_name

        contact_store_path = os.path.join(work_dir, "contact_information.json")

        log_buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(log_buffer):
                ranked = cvscreen.run_batch(job_path, cv_paths, contact_store_path=contact_store_path)
        except Exception as exc:
            print(f"[/screen] run_batch failed: {exc}")
            return jsonify(
                success=False,
                error=(
                    f"The screening pipeline raised an error: {exc}. If that mentions a "
                    "connection problem, check that Ollama is running locally on port 11434 "
                    "with the granite4.1:3b model pulled."
                ),
            ), 500

        contact_info = {}
        if os.path.exists(contact_store_path):
            with open(contact_store_path, "r", encoding="utf-8") as f:
                contact_info = json.load(f)

        for candidate in ranked:
            candidate["source_file"] = original_names.get(
                candidate["source_file"], os.path.basename(candidate["source_file"])
            )

        return jsonify(
            success=True,
            candidates=ranked,
            contact_info=contact_info,
            notes=_extract_notes(log_buffer.getvalue(), original_names),
            uploaded_count=len(cv_files),
            ranked_count=len(ranked),
        )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    # Port 5001, not 5000 - 5000 collides with macOS AirPlay Receiver.
    app.run(debug=True, port=5001)