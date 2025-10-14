const BASE_URL = "http://127.0.0.1:8000";
const locationSelect = document.getElementById("location");
const analyzeBtn = document.getElementById("analyzeBtn");
const downloadBtn = document.getElementById("downloadBtn");
const summaryDiv = document.getElementById("summary");

let historyChart, predictionChart;

// Analyze Button
analyzeBtn.addEventListener("click", async () => {
  const location = locationSelect.value.trim();
  if (!location) {
    alert("Please select a location first!");
    return;
  }

  try {
    const [dataRes, summaryRes] = await Promise.all([
      fetch(`${BASE_URL}/api/data/${location}`),
      fetch(`${BASE_URL}/api/summary/${location}`)
    ]);

    if (!dataRes.ok || !summaryRes.ok) throw new Error("Failed to fetch data");

    const data = await dataRes.json();
    const summaryData = await summaryRes.json();

    renderCharts(data.history, summaryData.future);
    renderSummary(summaryData.summary);

  } catch (error) {
    console.error("Error:", error);
  }
});

// Render summary (50-word style)
function renderSummary(summary) {
  summaryDiv.classList.remove("hidden");
  summaryDiv.innerHTML = `
    <h3>📊 Summary</h3>
    <p>${summary.text}</p>
    <p><b>Average Rate:</b> ${summary.average_rate}%</p>
    <p><b>Min Rate:</b> ${summary.min_rate}% | <b>Max Rate:</b> ${summary.max_rate}%</p>
    <h4>🌍 Causes:</h4>
    <ul>${summary.causes.map(c => `<li>${c}</li>`).join('')}</ul>
    <h4>🌱 Reduction Methods:</h4>
    <ul>${summary.reduction_methods.map(r => `<li>${r}</li>`).join('')}</ul>
  `;
}

// Render charts
function renderCharts(history, future) {
  const ctx1 = document.getElementById("historyChart").getContext("2d");
  const ctx2 = document.getElementById("predictionChart").getContext("2d");

  const years = history.map(row => row.Year);
  const rates = history.map(row => parseFloat(row["Deforestation_Rate_%"]));

  const futureYears = future.years;
  const futureRates = future.rates;

  if (historyChart) historyChart.destroy();
  if (predictionChart) predictionChart.destroy();

  // Historical chart
  historyChart = new Chart(ctx1, {
    type: "line",
    data: {
      labels: years,
      datasets: [{
        label: "Deforestation Rate (%)",
        data: rates,
        borderColor: "#c62828",
        backgroundColor: "rgba(198, 40, 40, 0.1)",
        tension: 0.3
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { display: true } }
    }
  });

  // Prediction chart
  predictionChart = new Chart(ctx2, {
    type: "line",
    data: {
      labels: futureYears,
      datasets: [{
        label: "Predicted Rate (%)",
        data: futureRates,
        borderColor: "#2e7d32",
        backgroundColor: "rgba(46, 125, 50, 0.1)",
        tension: 0.3
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { display: true } }
    }
  });
}

// Download Report
downloadBtn.addEventListener("click", async () => {
  const location = locationSelect.value.trim();
  if (!location) return alert("Select a location first!");

  try {
    const response = await fetch(`${BASE_URL}/api/report/${location}`);
    if (!response.ok) throw new Error("Failed to download report");

    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `Deforestation_Report_${location}.pdf`;
    a.click();
  } catch (error) {
    console.error("Download error:", error);
  }
});
