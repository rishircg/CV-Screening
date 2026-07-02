(() => {
  "use strict";

  const CATEGORIES = [
    { key: "experiences", label: "Experiences", weight: 30 },
    { key: "skills", label: "Skills", weight: 25 },
    { key: "projects", label: "Projects", weight: 20 },
    { key: "education", label: "Education", weight: 15 },
    { key: "extra_curricular", label: "Extra Curricular", weight: 10 },
  ];

  const LOADING_MESSAGES = [
    "Reading files…",
    "Checking CVs are valid…",
    "Redacting personal information…",
    "Classifying CV content…",
    "Scoring against the job description…",
    "Ranking candidates…",
  ];
  const LOADING_STEADY_MESSAGE = "Still working — larger batches take longer…";
  const LOADING_STEP_MS = 4500;

  const form = document.getElementById("intake-form");
  const cvInput = document.getElementById("cv-input");
  const cvFileList = document.getElementById("cv-file-list");
  const submitBtn = document.getElementById("submit-btn");

  const intakePanel = document.getElementById("intake-panel");
  const loadingPanel = document.getElementById("loading-panel");
  const loadingStatus = document.getElementById("loading-status");
  const resultsPanel = document.getElementById("results-panel");
  const errorBanner = document.getElementById("error-banner");

  const resultsCount = document.getElementById("results-count");
  const notesPanel = document.getElementById("notes-panel");
  const notesList = document.getElementById("notes-list");
  const candidateList = document.getElementById("candidate-list");

  const downloadResultsBtn = document.getElementById("download-results-btn");
  const downloadContactsBtn = document.getElementById("download-contacts-btn");
  const resetBtn = document.getElementById("reset-btn");

  let loadingTimer = null;
  let lastCandidates = [];
  let lastContactInfo = {};

  function escapeHtml(value) {
    const div = document.createElement("div");
    div.textContent = value ?? "";
    return div.innerHTML;
  }

  function showPanel(panel) {
    for (const p of [intakePanel, loadingPanel, resultsPanel]) {
      p.hidden = p !== panel;
    }
  }

  function showError(message) {
    errorBanner.textContent = message;
    errorBanner.hidden = false;
  }

  function clearError() {
    errorBanner.hidden = true;
    errorBanner.textContent = "";
  }

  cvInput.addEventListener("change", () => {
    const files = Array.from(cvInput.files || []);
    if (files.length === 0) {
      cvFileList.textContent = "";
      return;
    }
    const label = files.length === 1 ? "1 file selected" : `${files.length} files selected`;
    cvFileList.textContent = `${label}: ${files.map((f) => f.name).join(", ")}`;
  });

  function startLoadingMessages() {
    let i = 0;
    loadingStatus.textContent = LOADING_MESSAGES[0];
    loadingTimer = setInterval(() => {
      i += 1;
      loadingStatus.textContent =
        i < LOADING_MESSAGES.length ? LOADING_MESSAGES[i] : LOADING_STEADY_MESSAGE;
    }, LOADING_STEP_MS);
  }

  function stopLoadingMessages() {
    if (loadingTimer) {
      clearInterval(loadingTimer);
      loadingTimer = null;
    }
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    clearError();

    const formData = new FormData(form);

    submitBtn.disabled = true;
    showPanel(loadingPanel);
    startLoadingMessages();

    let response;
    let data;
    try {
      response = await fetch("/screen", { method: "POST", body: formData });
      data = await response.json();
    } catch (err) {
      stopLoadingMessages();
      submitBtn.disabled = false;
      showPanel(intakePanel);
      showError(
        "Couldn't reach the server. Check that the Flask app is still running, then try again."
      );
      return;
    }

    stopLoadingMessages();
    submitBtn.disabled = false;

    if (!response.ok || !data.success) {
      showPanel(intakePanel);
      showError(data.error || "Something went wrong while screening these CVs.");
      return;
    }

    renderResults(data);
    showPanel(resultsPanel);
  });

  function renderResults(data) {
    lastCandidates = data.candidates || [];
    lastContactInfo = data.contact_info || {};

    resultsCount.textContent = `${data.ranked_count} of ${data.uploaded_count} CVs ranked`;

    if (data.notes && data.notes.length > 0) {
      notesList.innerHTML = data.notes.map((n) => `<li>${escapeHtml(n)}</li>`).join("");
      notesPanel.hidden = false;
    } else {
      notesList.innerHTML = "";
      notesPanel.hidden = true;
    }

    candidateList.innerHTML = lastCandidates.map(renderCandidateCard).join("");
  }

  function renderCandidateCard(candidate) {
    const rankClass = candidate.rank === 1 ? " rank-1" : "";
    const scores = candidate.category_scores || {};
    const contact = lastContactInfo[candidate.applicant_id];

    const categoryRows = CATEGORIES.map((cat) => {
      const entry = scores[cat.key] || { score: 0, justification: "" };
      const pct = Math.max(0, Math.min(100, Math.round(entry.score)));
      return `
        <div class="category-row">
          <div class="category-label">${escapeHtml(cat.label)}<span class="weight">${cat.weight}% weight</span></div>
          <div class="score-bar-row">
            <div class="score-bar"><div class="score-bar-fill" style="width:${pct}%"></div></div>
            <span class="category-score-num">${entry.score.toFixed(1)}</span>
          </div>
          <p class="category-justification">${escapeHtml(entry.justification)}</p>
        </div>`;
    }).join("");

    const contactBlock = contact
      ? `<div class="contact-block">
           <span class="contact-tag">For human review only</span>
           <p class="contact-text">${escapeHtml(contact)}</p>
         </div>`
      : "";

    return `
      <li class="candidate-card${rankClass}">
        <div class="candidate-head">
          <div class="candidate-badge">${escapeHtml(String(candidate.rank))}</div>
          <div class="candidate-id-block">
            <span class="candidate-id">Applicant ${escapeHtml(candidate.applicant_id)}</span>
            <p class="candidate-file">${escapeHtml(candidate.source_file)}</p>
          </div>
          <div class="candidate-score-block">
            <div class="candidate-score">${candidate.overall_score.toFixed(1)}</div>
            <div class="candidate-score-label">of 100</div>
          </div>
        </div>
        <p class="candidate-rationale">${escapeHtml(candidate.rationale)}</p>
        <button type="button" class="breakdown-toggle" aria-expanded="false">
          <span class="chev">▾</span> View full breakdown
        </button>
        <div class="breakdown" hidden>
          ${categoryRows}
          ${contactBlock}
        </div>
      </li>`;
  }

  candidateList.addEventListener("click", (event) => {
    const toggle = event.target.closest(".breakdown-toggle");
    if (!toggle) return;
    const card = toggle.closest(".candidate-card");
    const breakdown = card.querySelector(".breakdown");
    const expanded = toggle.getAttribute("aria-expanded") === "true";
    toggle.setAttribute("aria-expanded", String(!expanded));
    breakdown.hidden = expanded;
  });

  function downloadJson(filename, payload) {
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  downloadResultsBtn.addEventListener("click", () => {
    downloadJson("screening_results.json", lastCandidates);
  });

  downloadContactsBtn.addEventListener("click", () => {
    downloadJson("contact_information.json", lastContactInfo);
  });

  resetBtn.addEventListener("click", () => {
    form.reset();
    cvFileList.textContent = "";
    clearError();
    candidateList.innerHTML = "";
    showPanel(intakePanel);
  });
})();
