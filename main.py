from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
import docker

app = FastAPI()

# Mount the static directory
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
    result = {"status": "ok", "model": model}

    if model == "MLP":
        hidden_layers = data_dict.get("hidden_layers")
        neurons = data_dict.get("neurons")
        result["hidden_layers"] = hidden_layers
        result["neurons"] = neurons
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
        # Optionally, process the file content

    # Create a Docker container (example: using 'python:3.9-slim' image)
    client = docker.from_env()
    container = client.containers.run(
        "ubuntu:latest",  # Use your desired image
        command="ls",  # Dummy command, replace as needed
        detach=True,
        auto_remove=True,
    )
    result["container_id"] = container.id

    return JSONResponse(result)
