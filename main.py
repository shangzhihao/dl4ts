from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
import docker
from pathlib import Path
import json
import uuid
import mlflow


volume_path = Path(__file__).parent / "docker"
data_path = volume_path / "data"
docker_working_path = "/tl4ds"
docker_image = "shangzhihao/dl4ts:latest"
sample_file = "samples.csv"
train_script = "train.sh"
mlflow_dir = "mlflow_runs"
mlflow_exp_name = "dl4ts"
model_state_file = "model.pth"


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", include_in_schema=False)
@app.get("/index.html", include_in_schema=False)
async def index():
    return FileResponse("static/index.html")


@app.post("/train")
async def train(data: str = Form(...), file: UploadFile = File(None)):
    data_dict = json.loads(data)
    docker_envs = data_dict.copy()
    docker_envs["sample_file"] = sample_file
    docker_envs["mlflow_dir"] = mlflow_dir
    docker_envs["mlflow_exp_name"] = mlflow_exp_name
    docker_envs["model_state_file"] = model_state_file
    job_id = str(uuid.uuid4())
    # create a unique job path
    job_path = data_path / job_id
    job_path.mkdir(parents=True, exist_ok=False)
    # Save uploaded file directly to job_path
    if file is not None:
        file_path = job_path / sample_file
        with open(file_path, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)

    docker_envs["work_dir"] = docker_working_path
    docker_envs["job_id"] = job_id
    client = docker.from_env()
    container = client.containers.create(
        docker_image,
        command=["sh", f"{docker_working_path}/{train_script}"],
        detach=True,
        auto_remove=False,
        volumes={str(volume_path): {"bind": docker_working_path, "mode": "rw"}},
        environment=docker_envs,
    )
    docker_envs["cid"] = container.id
    docker_envs["job_id"] = job_id
    container.start()
    return JSONResponse(docker_envs)


@app.get("/status/{job_id}")
async def status(job_id: str):
    mlflow_path = data_path / job_id / mlflow_dir
    uri = f"file://{mlflow_path}"
    mlflow.set_tracking_uri(uri)
    client = mlflow.MlflowClient(tracking_uri=uri)
    exp = client.get_experiment_by_name(mlflow_exp_name)
    if not exp:
        return JSONResponse({"error": "Experiment not found"}, status_code=404)
    runs = client.search_runs(experiment_ids=[exp.experiment_id])
    if not runs:
        return JSONResponse({"error": "No runs found for the experiment"}, status_code=404)
    run = runs[0]
    train_loss = client.get_metric_history(run.info.run_id, "train_loss")
    val_loss = client.get_metric_history(run.info.run_id, "val_loss")
    progress = client.get_metric_history(run.info.run_id, "progress")
    result = {
        "epochs": run.data.params.get("epochs", "N/A"),
        "progress": [m.value for m in progress][:-1],
        "train_loss": [m.value for m in train_loss][:-1],
        "val_loss": [m.value for m in val_loss][:-1],}
    return JSONResponse(result)