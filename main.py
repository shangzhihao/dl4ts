from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
import docker
from pathlib import Path


volume_path = Path(__file__).parent / "docker"
docker_working_path = "/tl4ds"

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", include_in_schema=False)
@app.get("/index.html", include_in_schema=False)
async def favicon():
    return FileResponse("static/index.html")


@app.post("/train")
async def train(data: str = Form(...), file: UploadFile = File(None)):
    import json

    data_dict = json.loads(data)
    model = data_dict.get("model")
    result = {}

    if model == "MLP":
        hidden_layers = data_dict.get("hidden_layers")
        neurons = data_dict.get("neurons")
        activation_function = data_dict.get("activation_function")
        input_window = data_dict.get("input_window")
        batch_size = data_dict.get("batch_size")
        epochs = data_dict.get("epochs")
        learning_rate = data_dict.get("learning_rate")
    elif model == "xLSTM":
        # Handle xLSTM parameters
        pass
    elif model == "Transformer":
        # Handle Transformer parameters
        pass

    # Handle uploaded file
    if file:
        file_content = await file.read()
        result["file_name"] = file.filename
        result["file_size"] = len(file_content)

    # Prepare environment variables for Docker
    env_vars = {"MODEL": model, "WDIR": docker_working_path}
    for k, v in data_dict.items():
        if v is not None:
            env_vars[k.upper()] = str(v)

    client = docker.from_env()
    container = client.containers.create(
        "ubuntu:latest",
        command=["sh", f"{docker_working_path}/train.sh"],
        detach=True,
        auto_remove=False,
        volumes={str(volume_path): {"bind": docker_working_path, "mode": "rw"}},
        environment=env_vars,
    )
    result["job_id"] = container.id
    container.start()
    return JSONResponse(result)
