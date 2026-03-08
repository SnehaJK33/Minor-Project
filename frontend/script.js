const BASE_URL = "/api";

const locationSelect = document.getElementById("location");
const analyzeBtn = document.getElementById("analyzeBtn");
const downloadBtn = document.getElementById("downloadBtn");

const summaryDiv = document.getElementById("summary");
let historyChart, predictionChart, gasChart, fireChart, gainChart;

// -------------------------------------
// ✅ Convert % , comma values into number
// Example: "10.30%" -> 10.30
// Example: "1,740,000" -> 1740000
// -------------------------------------
function toNumber(value) {
  if (value === null || value === undefined) return 0;

  value = value.toString().trim();
  value = value.replace(/,/g, "").replace(/%/g, "");

  let num = parseFloat(value);
  return isNaN(num) ? 0 : num;
}

function toNullableNumber(value) {
  if (value === null || value === undefined) return null;
  const cleaned = value.toString().trim().replace(/,/g, "").replace(/%/g, "");
  const num = parseFloat(cleaned);
  return Number.isFinite(num) ? num : null;
}

function avgOf(values) {
  const valid = values.filter(v => Number.isFinite(v));
  if (!valid.length) return null;
  return valid.reduce((a, b) => a + b, 0) / valid.length;
}

function normalizeNonNegativeRates(values = []) {
  const nums = values.map(v => toNullableNumber(v)).filter(Number.isFinite);
  if (!nums.length) return [];
  const allNonPositive = nums.every(v => v <= 0);
  const hasNegative = nums.some(v => v < 0);
  if (allNonPositive && hasNegative) return nums.map(v => Number(Math.abs(v).toFixed(2)));
  return nums.map(v => Math.max(0, Number(v.toFixed(2))));
}

function inferInsightsFromHistory(history = []) {
  const hasNumber = (v) => Number.isFinite(v);
  const rates = history.map(r => toNullableNumber(r["Deforestation_Rate_%"])).filter(hasNumber);
  const urban = history.map(r => toNullableNumber(r["Urbanization Rate (%)"] ?? r["Urbanization Rate"])).filter(hasNumber);
  const aqi = history.map(r => toNullableNumber(r["Pollution Index (AQI)"] ?? r["Pollution_Index"])).filter(hasNumber);
  const rain = history.map(r => toNullableNumber(r["Rainfall_mm"] ?? r["Rainfall (mm)"])).filter(hasNumber);

  let slope = 0;
  if (rates.length >= 2) slope = rates[rates.length - 1] - rates[0];

  const causes = [];
  if (slope > 0.8) causes.push("Rising year-on-year forest pressure and land conversion");
  if (avgOf(urban) !== null && avgOf(urban) > 20) causes.push("Urban expansion and infrastructure growth");
  if (avgOf(aqi) !== null && avgOf(aqi) > 80) causes.push("Industrial and pollution-linked ecological stress");
  if (avgOf(rain) !== null && avgOf(rain) < 1200) causes.push("Lower rainfall affecting natural forest recovery");
  if (!causes.length) causes.push("Mixed anthropogenic pressure (settlement, extraction, and land-use change)");

  const reductionMethods = [
    "Community-led afforestation and degraded-land restoration",
    "Satellite-based monitoring and strict anti-encroachment enforcement",
    "Sustainable agriculture/forestry practices in buffer zones",
    "District-level conservation planning with yearly compliance audits"
  ];

  return { causes, reductionMethods };
}

function solveLinear3x3(A, b) {
  const m = [
    [A[0][0], A[0][1], A[0][2], b[0]],
    [A[1][0], A[1][1], A[1][2], b[1]],
    [A[2][0], A[2][1], A[2][2], b[2]]
  ];

  for (let i = 0; i < 3; i++) {
    let pivot = i;
    for (let r = i + 1; r < 3; r++) {
      if (Math.abs(m[r][i]) > Math.abs(m[pivot][i])) pivot = r;
    }
    if (Math.abs(m[pivot][i]) < 1e-12) return null;
    if (pivot !== i) {
      const tmp = m[i];
      m[i] = m[pivot];
      m[pivot] = tmp;
    }

    const div = m[i][i];
    for (let c = i; c < 4; c++) m[i][c] /= div;

    for (let r = 0; r < 3; r++) {
      if (r === i) continue;
      const factor = m[r][i];
      for (let c = i; c < 4; c++) m[r][c] -= factor * m[i][c];
    }
  }

  return [m[0][3], m[1][3], m[2][3]];
}

function fallbackFutureFromHistory(history = [], nYears = 5) {
  const rows = history
    .map(r => ({
      year: toNullableNumber(r.Year),
      rate: toNullableNumber(r["Deforestation_Rate_%"])
    }))
    .filter(r => Number.isFinite(r.year) && Number.isFinite(r.rate))
    .sort((a, b) => a.year - b.year);

  if (rows.length < 2) return { years: [], rates: [] };

  const x = rows.map(r => r.year - rows[0].year);
  const y = rows.map(r => r.rate);
  const n = x.length;

  const s1 = x.reduce((a, v) => a + v, 0);
  const s2 = x.reduce((a, v) => a + v * v, 0);
  const s3 = x.reduce((a, v) => a + v * v * v, 0);
  const s4 = x.reduce((a, v) => a + v * v * v * v, 0);
  const sy = y.reduce((a, v) => a + v, 0);
  const sxy = x.reduce((a, v, i) => a + v * y[i], 0);
  const sx2y = x.reduce((a, v, i) => a + v * v * y[i], 0);

  let coeff = null;
  if (n >= 3) {
    coeff = solveLinear3x3(
      [
        [n, s1, s2],
        [s1, s2, s3],
        [s2, s3, s4]
      ],
      [sy, sxy, sx2y]
    );
  }

  const last = rows[rows.length - 1];
  let c0 = last.rate;
  let c1 = 0;
  let c2 = 0;

  if (coeff) {
    // y = c0 + c1*x + c2*x^2
    c0 = coeff[0];
    c1 = coeff[1];
    c2 = coeff[2];
  } else {
    const dy = rows[rows.length - 1].rate - rows[rows.length - 2].rate;
    const dx = rows[rows.length - 1].year - rows[rows.length - 2].year || 1;
    c1 = dy / dx;
  }

  const years = [];
  const rawRates = [];
  for (let i = 1; i <= nYears; i++) {
    const nextYear = last.year + i;
    const nx = nextYear - rows[0].year;
    const pred = c0 + c1 * nx + c2 * nx * nx;
    years.push(nextYear);
    rawRates.push(pred);
  }

  const rates = normalizeNonNegativeRates(rawRates);
  return { years, rates };
}

function normalizeFuturePrediction(future, history) {
  const years = Array.isArray(future?.years) ? future.years.map(y => toNullableNumber(y)).filter(Number.isFinite) : [];
  const rates = Array.isArray(future?.rates)
    ? normalizeNonNegativeRates(future.rates)
    : [];

  let normalized = { years, rates };
  const invalid = !normalized.years.length || !normalized.rates.length || normalized.years.length !== normalized.rates.length;

  if (invalid) return { ...fallbackFutureFromHistory(history, 5), used_fallback: true };

  const maxRate = Math.max(...normalized.rates);
  const minRate = Math.min(...normalized.rates);
  const span = maxRate - minRate;
  const isFlat = span < 0.2;
  const isAllZero = normalized.rates.every(r => Math.abs(r) < 1e-9);

  if (isFlat || isAllZero) return { ...fallbackFutureFromHistory(history, 5), used_fallback: true };

  return { ...normalized, used_fallback: false };
}

// -------------------------------------
// Safe Fetch Function (No JSON Crash)
// -------------------------------------
async function fetchJSON(url) {
  const res = await fetch(url);

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API Error (${res.status}): ${text}`);
  }

  return await res.json();
}

// -------------------------------------
// ANALYZE BUTTON CLICK
// -------------------------------------
analyzeBtn.addEventListener("click", async () => {
  const location = locationSelect.value.trim();

  if (!location) {
    alert("Please select a district first!");
    return;
  }

  try {
    const dataJson = await fetchJSON(`${BASE_URL}/data/${encodeURIComponent(location)}`);
    const summaryJson = await fetchJSON(`${BASE_URL}/summary/${encodeURIComponent(location)}`);
    let gainJson = { gain_ha: 0 };
    try {
      gainJson = await fetchJSON(`${BASE_URL}/district/gain/${encodeURIComponent(location)}`);
    } catch (gainErr) {
      console.warn("Gain dataset missing for district, fallback to 0:", gainErr.message);
    }
    const history = dataJson.history || [];
    // Deduplicate history by Year (keep last occurrence) and sort by year
    function dedupeByYear(rows) {
      const m = new Map();
      rows.forEach(r => {
        const y = toNumber(r.Year);
        if (!Number.isFinite(y)) return;
        m.set(y, r);
      });
      return Array.from(m.entries()).sort((a,b) => a[0] - b[0]).map(e => e[1]);
    }

    const dedupHistory = dedupeByYear(history);
    const summary = summaryJson.summary || {};
    let future = summaryJson.future_prediction_5_years || { years: [], rates: [] };
    const predictionMeta = {
      modelName: future.model_name || "Polynomial Regression (degree=2)",
      r2: toNullableNumber(future.model_accuracy_r2 ?? future.r2_score),
      trainingSamples: toNullableNumber(future.training_samples)
    };
    const environment = summaryJson.environment || {};
    const risk = summaryJson.risk_assessment || [];

    if (dedupHistory.length === 0) {
      alert("No historical data found for this district!");
      return;
    }

    future = normalizeFuturePrediction(future, dedupHistory);
    predictionMeta.usedFallback = !!future.used_fallback;

    renderSummary(summary, environment, risk, location, dedupHistory, predictionMeta);
    renderCharts(dedupHistory, future, gainJson.gain_ha ?? 0);

  } catch (error) {
    console.error("❌ Error:", error);
    alert("Something went wrong! Check backend + console.\n\n" + error.message);
  }
});

// -------------------------------------
// DOWNLOAD PDF BUTTON
// -------------------------------------
downloadBtn.addEventListener("click", () => {
  const location = locationSelect.value.trim();

  if (!location) {
    alert("Select district first!");
    return;
  }

  window.open(`${BASE_URL}/report/${encodeURIComponent(location)}`, "_blank");
});

// -------------------------------------
// SUMMARY SECTION (UPDATED)
// -------------------------------------
function renderSummary(summary, environment, risk, location, history = [], predictionMeta = {}) {
  summaryDiv.style.display = "block";
  const hasNumber = (v) => Number.isFinite(v);
  const rateSeries = history.map(r => toNullableNumber(r["Deforestation_Rate_%"])).filter(hasNumber);
  const rainfallSeries = history.map(r => toNullableNumber(r["Rainfall_mm"] ?? r["Rainfall (mm)"])).filter(hasNumber);
  const tempSeries = history.map(r => toNullableNumber(r["Temperature_C"] ?? r["Temperature (°C)"] ?? r["Temperature (Â°C)"])).filter(hasNumber);
  const aqiSeries = history.map(r => toNullableNumber(r["Pollution Index (AQI)"] ?? r["Pollution_Index"])).filter(hasNumber);
  const urbanSeries = history.map(r => toNullableNumber(r["Urbanization Rate (%)"] ?? r["Urbanization Rate"] ?? r["Urbanisation Rate (%)"])).filter(hasNumber);
  const popSeries = history.map(r => toNullableNumber(r["Population (Est.)"] ?? r["Population"] ?? r["Population (Estimated)"])).filter(hasNumber);

  // Use toNumber helper to avoid NaN issues
  const avgRate = toNullableNumber(
    summary.average_rate ??
    summary.averageRate ??
    summary.average_deforestation_rate ??
    avgOf(rateSeries)
  );
  const minRate = toNullableNumber(
    summary.min_rate ??
    summary.minRate ??
    summary.min_deforestation_rate ??
    (rateSeries.length ? Math.min(...rateSeries) : null)
  );
  const maxRate = toNullableNumber(
    summary.max_rate ??
    summary.maxRate ??
    summary.max_deforestation_rate ??
    (rateSeries.length ? Math.max(...rateSeries) : null)
  );

  const inferredInsights = inferInsightsFromHistory(history);
  const causesList = (summary.causes || summary.Causes || inferredInsights.causes || []).filter(Boolean);
  const reductionList = (summary.reduction_methods || summary.reductionMethods || inferredInsights.reductionMethods || []).filter(Boolean);
  const causesHtml = causesList.map(c => `<li>${c}</li>`).join("") || "<li>No data</li>";
  const reductionHtml = reductionList.map(r => `<li>${r}</li>`).join("") || "<li>No data</li>";

  const envTemp = toNullableNumber(
    environment.avg_temp ??
    environment.avgTemp ??
    environment.average_temperature_c ??
    avgOf(tempSeries)
  );
  const envRain = toNullableNumber(
    environment.avg_rainfall ??
    environment.avgRainfall ??
    environment.average_rainfall_mm ??
    avgOf(rainfallSeries)
  );
  const envAq = toNullableNumber(
    environment.avg_pollution ??
    environment.avgPollution ??
    avgOf(aqiSeries)
  );
  const envUrban = toNullableNumber(
    environment.avg_urbanization ??
    environment.avgUrbanization ??
    avgOf(urbanSeries)
  );
  const envPop = toNullableNumber(
    environment.avg_population ??
    environment.avgPopulation ??
    avgOf(popSeries)
  );

  // Handle both list-based risk entries and object-based risk response
  let riskHtml = "<li>No risk data</li>";
  if (risk && !Array.isArray(risk) && risk.risk_level) {
    riskHtml = `<li><b>Risk Level:</b> ${risk.risk_level}</li>`;
  } else if ((risk || []).length) {
    const rmap = new Map();
    (risk || []).forEach(r => {
      const y = r.Year != null ? Number(r.Year) : (r.year != null ? Number(r.year) : null);
      if (y !== null && !Number.isNaN(y)) {
        rmap.set(y, r);
      }
    });
    const uniq = Array.from(rmap.entries()).sort((a,b)=>a[0]-b[0]).map(e=>e[1]);
    if (uniq.length) {
      riskHtml = uniq.map(r => ` 
        <li><b>${r.Year ?? r.year ?? 'N/A'}</b> → Loss: ${r.Forest_Loss ?? r.forestLoss ?? 'N/A'} sq.km → <b>${r.Risk_Level ?? r.riskLevel ?? 'Unknown'}</b></li>
      `).join("");
    }
  }

  summaryDiv.innerHTML = `
    <h2>📌 Summary Report - ${location}</h2>

    <p><b>Average Rate:</b> ${hasNumber(avgRate) ? avgRate.toFixed(2) + '%' : 'N/A'}</p>
    <p><b>Min Rate:</b> ${hasNumber(minRate) ? minRate.toFixed(2) + '%' : 'N/A'}</p>
    <p><b>Max Rate:</b> ${hasNumber(maxRate) ? maxRate.toFixed(2) + '%' : 'N/A'}</p>

    <h3>🌍 Causes</h3>
    <ul>${causesHtml}</ul>

    <h3>🌱 Reduction Methods</h3>
    <ul>${reductionHtml}</ul>

    <h3>🤖 Prediction Model</h3>
    <p><b>Model:</b> ${predictionMeta.modelName || "Polynomial Regression (degree=2)"}</p>
    <p><b>Accuracy (R²):</b> ${hasNumber(predictionMeta.r2) ? predictionMeta.r2.toFixed(3) : "N/A"}</p>
    <p><b>Training Samples:</b> ${hasNumber(predictionMeta.trainingSamples) ? Math.round(predictionMeta.trainingSamples) : "N/A"}</p>
    <p><b>Prediction Source:</b> ${predictionMeta.usedFallback ? "Frontend fallback trend (used due to flat/invalid API forecast)" : "Backend model output"}</p>

    <h3>⚠️ Risk Assessment (Clustering)</h3>
    <ul>${riskHtml}</ul>
  `;
}

// -------------------------------------
// TABLE SECTION
// -------------------------------------
function renderTables() {}

// -------------------------------------
// CHARTS SECTION
// -------------------------------------
function renderCharts(history, future, gainHa = 0) {
  const years = history.map(r => toNumber(r.Year));

  const defRates = history.map(r => toNumber(r["Deforestation_Rate_%"]));
  const gas = history.map(r => toNumber(r["Green-House_Gases"] ?? r["gfw_gross_emissions_co2e_all_gases__Mg"]));
  const fire = history.map(r => toNumber(r["Tree_cover_loss_from_fires"] ?? 0));
  // Destroy old charts
  if (historyChart) historyChart.destroy();
  if (predictionChart) predictionChart.destroy();
  if (gasChart) gasChart.destroy();
  if (fireChart) fireChart.destroy();
  if (gainChart) gainChart.destroy();

  // -------------------------------
  // 1️⃣ History Chart
  // -------------------------------
  historyChart = new Chart(document.getElementById("historyChart"), {
    type: "line",
    data: {
      labels: years,
      datasets: [{
        label: "Deforestation Rate (%)",
        data: defRates,
        borderColor: "green",
        borderWidth: 2,
        tension: 0.3
      }]
    }
  });

  // -------------------------------
  // 2️⃣ Prediction Chart
  // -------------------------------
  predictionChart = new Chart(document.getElementById("predictionChart"), {
    type: "line",
    data: {
      labels: future.years || [],
      datasets: [{
        label: "Predicted Rate (%)",
        data: (future.rates || []).map(v => toNumber(v)),
        borderColor: "red",
        borderWidth: 2,
        tension: 0.3
      }]
    }
  });

  // -------------------------------
  // 3️⃣ Rainfall Chart
  // -------------------------------
  gasChart = new Chart(document.getElementById("gasChart"), {
    type: "line",
    data: {
      labels: years,
      datasets: [{
        label: "Gas Emissions",
        data: gas,
        borderColor: "#1565c0",
        borderWidth: 2,
        tension: 0.3
      }]
    }
  });

  // -------------------------------
  // 4️⃣ Temperature Chart
  // -------------------------------
  fireChart = new Chart(document.getElementById("fireChart"), {
    type: "line",
    data: {
      labels: years,
      datasets: [{
        label: "Fire-Related Loss",
        data: fire,
        borderColor: "#d32f2f",
        borderWidth: 2,
        tension: 0.3
      }]
    }
  });

  // -------------------------------
  // 5️⃣ Gain Chart
  // -------------------------------
  const gainValue = toNumber(gainHa);
  gainChart = new Chart(document.getElementById("gainChart"), {
    type: "line",
    data: {
      labels: ["Start", "Overall Gain Forest"],
      datasets: [{
        label: "Gain (ha)",
        data: [0, gainValue],
        backgroundColor: "#43a04722",
        borderColor: "#2e7d32",
        pointBackgroundColor: "#2e7d32",
        pointRadius: 5,
        borderWidth: 2,
        tension: 0.2,
        fill: true
      }]
    },
    options: {
      scales: {
        y: { beginAtZero: true }
      }
    }
  });

}
