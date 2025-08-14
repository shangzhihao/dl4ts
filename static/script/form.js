fetch("/static/html/MLP.html")
  .then((response) => response.text())
  .then((data) => {
    const form = document.getElementById("mlp-form");
    form.innerHTML = data;
  });
