document.addEventListener("DOMContentLoaded", function () {
  // Chart.js setup for chart1 and chart2
  const ctx1 = document.getElementById("chart1").getContext("2d");
  const ctx2 = document.getElementById("chart2").getContext("2d");

  // Create empty charts
  const chart1 = new Chart(ctx1, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "Sample Data",
          data: [],
          borderColor: "rgba(75,192,192,1)",
          backgroundColor: "rgba(75,192,192,0.2)",
          fill: false,
          tension: 0.1,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      maintainAspectRatio: false,
      scales:{x:{grid:{display:false}},y:{grid:{display:false}}},
    },
  });

  const chart2 = new Chart(ctx2, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "Sample Data",
          data: [],
          borderColor: "rgba(255,99,132,1)",
          backgroundColor: "rgba(255,99,132,0.2)",
          fill: false,
          tension: 0.1,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      maintainAspectRatio: false,
      scales:{x:{grid:{display:false}},y:{grid:{display:false}}},
    },
  });

  // File picker event
  document
    .getElementById("sampleFileInput")
    .addEventListener("change", function (event) {
      const file = event.target.files[0];
      if (!file) return;

      const reader = new FileReader();
      reader.onload = function (e) {
        const text = e.target.result;
        const floatArray = text
          .split("\n")
          .map((line) => line.trim())
          .filter((line) => line.length > 0)
          .map(Number)
          .filter((num) => !isNaN(num));

        // Update chart1 and chart2 with the floatArray
        const labels = floatArray.map((_, i) => i + 1);

        chart1.data.labels = labels;
        chart1.data.datasets[0].data = floatArray;
        chart1.update();

        chart2.data.labels = labels;
        chart2.data.datasets[0].data = floatArray;
        chart2.update();
      };
      reader.readAsText(file);
    });
});
