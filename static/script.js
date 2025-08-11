document.addEventListener("DOMContentLoaded", function () {
  // Chart.js setup for chart1
  const chart1Canvas = document.getElementById("chart1");
  const ctx1 = chart1Canvas.getContext("2d");

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

  // File picker event
  document
    .getElementById("samples")
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

        // Update chart with the floatArray
        const labels = floatArray.map((_, i) => i + 1);

        chart1.data.labels = labels;
        chart1.data.datasets[0].data = floatArray;
        chart1.update();
      };
      reader.readAsText(file);
    });

  // Model select form toggling (with forms)
  const modelSelect = document.getElementById("modelSelect");
  const forms = {
    MLP: document.getElementById("mlp-form"),
    xLSTM: document.getElementById("xlstm-form"),
    Transformer: document.getElementById("transformer-form"),
  };

  function showForm(model) {
    Object.keys(forms).forEach((key) => {
      forms[key].style.display = key === model ? "block" : "none";
    });
  }

  // Initial display
  showForm(modelSelect.value);

  modelSelect.addEventListener("change", function () {
    showForm(this.value);
  });

  document.getElementById("train-btn").addEventListener("click", function () {
    const model = document.getElementById("modelSelect").value;
    let data = { model };

    if (model === "MLP") {
      data.mlp_hidden_layers = document.getElementById("mlp-hidden-layers").value;
      data.mlp_neurons = document.getElementById("mlp-neurons").value;
      data.mlp_act_fun = document.getElementById("mlp-act-fun").value;
      data.mlp_window = document.getElementById("mlp-window").value;
      data.mlp_batch = document.getElementById("mlp-batch").value;
      data.mlp_epochs = document.getElementById("mlp-epochs").value;
      data.mlp_lr = document.getElementById("mlp-lr").value;
    } else if (model === "xLSTM") {
      // Add xLSTM form data here if available
    } else if (model === "Transformer") {
      // Add Transformer form data here if available
    }

    // Get the selected file
    const fileInput = document.getElementById("samples");
    const file = fileInput.files[0];

    // Use FormData to send both JSON and file
    const formData = new FormData();
    formData.append("data", JSON.stringify(data));
    if (file) {
      formData.append("file", file);
    }

    fetch("/train", {
      method: "POST",
      body: formData,
    })
      .then((response) => response.json())
      .then((result) => {
        console.log("Train result:", result);
        // Handle result as needed
      })
      .catch((error) => {
        console.error("Error:", error);
      });
  });
});
