function fetchForm(model, formId) {
  fetch(`/static/html/${model}.html`)
    .then((response) => response.text())
    .then((data) => {
      const form = document.getElementById(formId);
      form.innerHTML = data;
    })
    .catch((error) => {
      console.error("Error fetching form:", error);
    });
}

fetchForm("MLP", "mlp-form");
fetchForm("LSTM", "lstm-form");
fetchForm("xLSTM", "xlstm-form");
fetchForm("Transformer", "transformer-form");
fetchForm("Trainer", "trainer");