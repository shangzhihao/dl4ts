from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
import docker
from pathlib import Path
import json
import uuid

volume_path = Path(__file__).parent / "docker"
data_path = volume_path / "data"
docker_working_path = "/tl4ds"

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", include_in_schema=False)
@app.get("/index.html", include_in_schema=False)
async def favicon():
    return FileResponse("static/index.html")


@app.post("/train")
async def train(data: str = Form(...), file: UploadFile = File(None)):
    data_dict = json.loads(data)
    result = data_dict.copy()
    job_id = str(uuid.uuid4())
    # create a unique job path
    job_path = data_path / job_id
    job_path.mkdir(parents=True, exist_ok=False)
    # Save uploaded file directly to job_path
    if file is not None:
        file_path = job_path / "samples.csv"
        with open(file_path, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        result["file"] = str(file_path)

    result["work_dir"] = docker_working_path
    result["job_id"] = job_id
    client = docker.from_env()
    container = client.containers.create(
        "ubuntu:latest",
        command=["sh", f"{docker_working_path}/train.sh"],
        detach=True,
        auto_remove=False,
        volumes={str(volume_path): {"bind": docker_working_path, "mode": "rw"}},
        environment=result,
    )
    result["cid"] = container.id
    result["job_id"] = job_id
    container.start()
    return JSONResponse(result)
