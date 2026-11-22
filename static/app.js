const textInput = document.getElementById("textInput");
const predictBtn = document.getElementById("predictBtn");
const errorBox = document.getElementById("errorBox");

// Sidebar navigation
document.querySelectorAll(".nav-item").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".nav-item").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".section").forEach((s) => s.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(`sec-${btn.dataset.section}`).classList.add("active");
  });
});

// Sample chips
document.querySelectorAll(".sample-chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    textInput.value = chip.dataset.text;
    predict();
  });
});

predictBtn.addEventListener("click", predict);
textInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") predict();
});

async function predict() {
  const text = textInput.value.trim();
  if (!text) return;

  predictBtn.disabled = true;
  errorBox.classList.add("hidden");

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Prediction failed.");
    renderPrediction(data);
  } catch (err) {
    errorBox.textContent = err.message;
    errorBox.classList.remove("hidden");
  } finally {
    predictBtn.disabled = false;
  }
}

function renderPrediction(data) {
  const meta = data.meta;
  const emojiEl = document.getElementById("resEmoji");
  const emoEl = document.getElementById("resEmotion");
  const fillEl = document.getElementById("confFill");

  emojiEl.textContent = meta.emoji;
  emojiEl.style.background = meta.bg || "#fef3c7";
  emoEl.textContent = meta.label;
  emoEl.style.color = meta.color;
  document.getElementById("resConf").textContent = `${data.confidence}%`;
  fillEl.style.width = `${data.confidence}%`;
  fillEl.style.background = meta.color;

  // Render token attention heatmap
  const tokenStrip = document.getElementById("tokenStrip");
  tokenStrip.innerHTML = data.tokens
    .map((t) => {
      const alpha = Math.max(0.08, t.norm * 0.78).toFixed(2);
      const textColor = t.norm > 0.55 ? "#ffffff" : "#1e1b4b";
      return `<span class="token-chip" style="background:rgba(79,70,229,${alpha});color:${textColor};" title="Attention: ${(t.norm * 100).toFixed(0)}%">${t.word}</span>`;
    })
    .join("");

  // Render live 6-emotion probabilities
  const liveProbs = document.getElementById("liveProbs");
  liveProbs.innerHTML = data.breakdown
    .map(
      (b) => `
      <div>
        <div class="lp-head">
          <span>${b.label}</span>
          <span style="font-family:'JetBrains Mono',monospace;">${b.score}%</span>
        </div>
        <div class="lp-track">
          <div class="lp-fill" style="width:${Math.max(b.score, 1)}%;background:${b.color};"></div>
        </div>
      </div>
    `
    )
    .join("");
}

async function initDashboard() {
  try {
    const res = await fetch("/api/dashboard");
    const data = await res.json();

    document.getElementById("sidebarModel").textContent = `Model: ${data.model_name}`;

    // Model Comparison bar chart
    const colors = ["#3b82f6", "#10b981", "#8b5cf6", "#312e81"];
    const chartEl = document.getElementById("modelChart");
    chartEl.innerHTML = (data.benchmark || [])
      .map((row, idx) => {
        const acc = Number(row["Accuracy"] || 0);
        const h = Math.max(20, Math.round(acc * 115));
        return `
          <div class="bar-col">
            <span class="bar-val">${acc.toFixed(2)}</span>
            <div class="bar-rect" style="height:${h}px;background:${colors[idx % colors.length]};"></div>
            <span class="bar-lbl">${row["Model"]}</span>
          </div>
        `;
      })
      .join("");

    // Per-Class list
    const pcEl = document.getElementById("perClassList");
    pcEl.innerHTML = (data.per_class || [])
      .map(
        (r) => `
        <div class="pc-row">
          <span class="pc-left"><span class="pc-dot" style="background:${r.color};"></span>${r.label}</span>
          <span class="pc-metrics">P: ${r.precision.toFixed(2)} &nbsp; R: ${r.recall.toFixed(2)}</span>
        </div>
      `
      )
      .join("");

    // Full Benchmark Table
    const tbody = document.getElementById("benchTableBody");
    tbody.innerHTML = (data.benchmark || [])
      .map(
        (r) => `
        <tr>
          <td><strong>${r["Model"]}</strong></td>
          <td>${(Number(r["Accuracy"] || 0) * 100).toFixed(2)}%</td>
          <td>${(Number(r["Macro F1"] || 0) * 100).toFixed(2)}%</td>
          <td>${(Number(r["Weighted F1"] || 0) * 100).toFixed(2)}%</td>
          <td>${(Number(r["Macro Precision"] || 0) * 100).toFixed(2)}%</td>
          <td>${(Number(r["Macro Recall"] || 0) * 100).toFixed(2)}%</td>
        </tr>
      `
      )
      .join("");
  } catch (e) {
    console.error("Dashboard load error:", e);
  }
}

initDashboard();
