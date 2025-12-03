# VisionTime

VisionTime is a GPU-enabled computer vision stack built around NVIDIA Triton Inference Server. It packages a Triton model repository, a lightweight orchestrator service, and an NGINX-hosted frontend into a single `docker compose` deployment.

## Prerequisites
- Docker with GPU support (NVIDIA Container Toolkit)
- Docker Compose plugin (`docker compose`)
- Internet access to download ONNX model weights (unless you provide a local model archive)

## Quickstart
You can bootstrap everything with the included helper script:

```bash
./quickstart.sh
```

This script:
1. Downloads the required ONNX model weights into `./models` using `scripts/download_models.py` (respects the `VISIONTIME_MODELS_URL` environment variable or `--source` argument for a prepacked tarball).
2. Builds the orchestrator image and starts all services defined in `docker-compose.yml` using `docker compose`.
3. Tails the combined container logs.

When the stack is running, services are exposed on the following ports:
- Triton HTTP/gRPC/metrics: 8000/8001/8002
- Orchestrator API: 8080
- Frontend (HTTPS via NGINX): 8090

To feed the pipeline with your laptop webcam, open the web UI and enable **Use this device's webcam** for a camera slot. The
browser will request camera permission and begin streaming frames directly into the orchestrator; you can mix browser feeds with
RTSP/HTTP streams by filling in URLs on the other rows.

## Populating models manually
If you prefer not to use `quickstart.sh`, you can invoke the downloader directly:

```bash
python scripts/download_models.py
```

To use a custom tar.gz bundle instead of downloading from Hugging Face, set `VISIONTIME_MODELS_URL` or pass `--source /path/to/models.tar.gz`. Use `--force` to overwrite existing model versions.

The `models` directory in the repository contains only configuration stubs by default; the ONNX weights are fetched at runtime.

## Troubleshooting
- **Triton exits with `failed to load all models`**: This usually means the ONNX model files have not been downloaded. Run `python scripts/download_models.py` (or `quickstart.sh`) to populate `./models/<model>/1/model.onnx` before starting the stack.
- **GPU access errors**: Ensure the NVIDIA Container Toolkit is installed and Docker is configured to expose GPUs to containers. The compose file requests `gpus: all` for Triton and the orchestrator.

## Useful commands
- `make build`: Build the orchestrator image.
- `make up`: Start the stack in the background.
- `make logs`: Tail logs from all services.
- `make down`: Stop and remove the containers.
