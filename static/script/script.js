let sampleChart, trainChart, valChart;
function makeChart(ctx) { 
  return new Chart(ctx, {
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
          pointRadius: 0
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
}


document.addEventListener("DOMContentLoaded", function () {
  // create a chart for the sample data
  const sampleCanvas= document.getElementById("sample-chart");
  const sampleCtx = sampleCanvas.getContext("2d");
  sampleChart = makeChart(sampleCtx);

  const trainCanvas= document.getElementById("train-loss-chart");
  const trainCtx = trainCanvas.getContext("2d");
  trainChart = makeChart(trainCtx);
  
  const valCanvas= document.getElementById("val-loss-chart");
  const valCtx = valCanvas.getContext("2d");
  valChart = makeChart(valCtx);
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
          .map((line) => line.split(",")[0])
          .filter((line) => line.length > 0)
          .map(Number)
          .filter((num) => !isNaN(num));

        // Update chart with the floatArray
        const labels = floatArray.map((_, i) => i + 1);
        sampleChart.data.labels = labels;
        sampleChart.data.datasets[0].data = floatArray;
        sampleChart.update();
      };
      reader.readAsText(file);
    });

  // Model select form toggling (with forms)
  const modelSelect = document.getElementById("modelSelect");
  const forms = {
    MLP: document.getElementById("mlp-form"),
    LSTM: document.getElementById("lstm-form"),
    TCN: document.getElementById("tcn-form"),
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

    // Map of model -> { inputId: payloadKey }
    const modelParamMap = {
      MLP: {
        "mlp-neurons": "mlp_neurons",
        "mlp-act-fun": "mlp_act_fun",
        "mlp-window": "mlp_window",
      },
      LSTM: {
        "lstm-layers": "lstm_layers",
        "lstm-dropout": "lstm_dropout",
        "lstm-window": "lstm_window",
        "lstm-hidden": "lstm_hidden",
      },
      TCN: {
        "tcn-window": "tcn_window",
        "tcn-kernel-size": "tcn_kernel_size",
        "tcn-channels": "tcn_channels",
        "tcn-dropout": "tcn_dropout",
      },
      Transformer: {
        "att-window": "att_window",
        "att-nhead": "att_nhead",
        "att-dmodel": "att_dmodel",
        "att-layers": "att_layers",
        "att-dropout": "att_dropout",
        "att-dim-forward": "att_dim_forward",
        "att-act-fun": "att_act_fun",
      },
    };

    // Collect model-specific params
    const paramMap = modelParamMap[model] || {};
    for (const [inputId, key] of Object.entries(paramMap)) {
      const el = document.getElementById(inputId);
      if (el) data[key] = el.value;
    }

    // Training parameters
    ["batch", "epochs", "lr", "optim", "scheduler"].forEach((id) => {
      const el = document.getElementById(id);
      if (el) data[id] = el.value;
    });
    const autoRadio = document.querySelector('input[name="auto"]:checked');
    data.auto = autoRadio ? autoRadio.value === "true" : false;
    const decay = document.querySelector('input[name="decay"]:checked');
    data.decay = decay ? decay.value === "true" : false;

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
        trackJob(result.job_id);
      })
      .catch((error) => {
        console.error("Error:", error);
      });
  });
});

function trackJob(jobId) {
    const intervalId = setInterval(() => {
        fetch(`/status/${jobId}`)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                console.log('Job tracking result:', data);
                train_loss = data.train_loss || [];
                val_loss = data.val_loss || [];
                progress = data.progress || [];
                epochs = parseInt(data.epochs, 10);
                // update trainChart with train_loss data
                if (train_loss.length > 0) {
                    const trainLabels = train_loss.map((_, i) => i + 1);
                    trainChart.data.labels = trainLabels;
                    trainChart.data.datasets[0].data = train_loss;
                    trainChart.data.datasets[0].label = "Train Loss";
                    trainChart.update();
                }
                
                // Update valChart with val_loss data
                if (val_loss.length > 0) {
                    const valLabels = val_loss.map((_, i) => i + 1);
                    valChart.data.labels = valLabels;
                    valChart.data.datasets[0].data = val_loss;
                    valChart.data.datasets[0].label = "Validation Loss";
                    valChart.update();
                }
                // update progress bar
                const pbar = document.getElementById("train-progress");
                pbar.style.display = "block";
                if(progress.length > 0){
                  const last = progress.length-1;
                  cur_epoch = progress[last] + 1
                  pbar.value = cur_epoch / epochs * 100;
                  if(cur_epoch == epochs){
                    clearInterval(intervalId);
                    console.log(`Job ${jobId} completed.`);
                  }
                }
            })
            .catch(error => {
                console.error('Error fetching job status:', error);
                clearInterval(intervalId);
            });
    }, 5000);
    
    console.log(`Started tracking job ${jobId} every 5 seconds`);
    return intervalId;
}
