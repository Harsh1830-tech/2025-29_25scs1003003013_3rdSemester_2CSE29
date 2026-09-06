// Spam Scanner — frontend logic.

const state = { model: "naive_bayes" };

const textarea = document.getElementById("email-text");
const reportBody = document.getElementById("report-body");

document.querySelectorAll(".model-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".model-btn").forEach(b => { b.classList.remove("active"); b.setAttribute("aria-checked", "false"); });
    btn.classList.add("active");
    btn.setAttribute("aria-checked", "true");
    state.model = btn.dataset.model;
  });
});

const EXAMPLES = {
  spam: "CONGRATULATIONS!!! You have WON $250000 in the International Lottery! To claim your prize, click the link below and enter your bank details within 24 hours. Offer expires in 24 hours, act fast!",
  ham: "Hi team, just a reminder that our project review is scheduled for Thursday at 2:30 PM. Please let me know if that time doesn't work for you. Thanks!",
};

document.querySelectorAll(".example-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    textarea.value = EXAMPLES[btn.dataset.kind];
    textarea.focus();
  });
});

function escapeHTML(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

async function scan() {
  const text = textarea.value.trim();
  if (!text) {
    reportBody.innerHTML = `<p class="error-text">Paste a message first — there's nothing to scan yet.</p>`;
    return;
  }

  reportBody.innerHTML = `<p class="placeholder">Scanning…</p>`;

  try {
    const res = await fetch("/api/classify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, model: state.model }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Something went wrong.");
    render(data);
  } catch (e) {
    reportBody.innerHTML = `<p class="error-text">${escapeHTML(e.message)}</p>`;
  }
}

function render(data) {
  const isSpam = data.label === "spam";
  const spamPct = Math.round((data.probabilities.spam || 0) * 100);
  const hamPct = Math.round((data.probabilities.ham || 0) * 100);

  const flaggedHTML = data.flagged_words.length
    ? `<p class="flag-title">Words that pushed this toward spam</p>
       <ul class="flag-words">${data.flagged_words.map(w => `<li>${escapeHTML(w)}</li>`).join("")}</ul>`
    : `<p class="flag-title">No strong spam-indicator words found in this message.</p>`;

  reportBody.innerHTML = `
    <span class="verdict ${isSpam ? "spam" : "ham"}">${isSpam ? "⚠ SPAM" : "✓ NOT SPAM"}</span>
    <div class="meter-row">
      <div class="meter-label"><span>Spam probability</span><span>${spamPct}%</span></div>
      <div class="meter-track"><div class="meter-fill spam" style="width:${spamPct}%"></div></div>
    </div>
    <div class="meter-row">
      <div class="meter-label"><span>Not-spam probability</span><span>${hamPct}%</span></div>
      <div class="meter-track"><div class="meter-fill ham" style="width:${hamPct}%"></div></div>
    </div>
    ${flaggedHTML}
  `;
}

document.getElementById("scan-btn").addEventListener("click", scan);
textarea.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") scan();
});
