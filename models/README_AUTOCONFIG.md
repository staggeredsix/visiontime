Triton Inference Server is launched with `strict_model_config=0`, allowing it to auto-generate model configurations from the ONNX files in this repository.

The previous hand-written `config.pbtxt` files for `clip_encoder`, `depth_fastdepth`, `optical_flow`, `rtmdet_s`, and `segformer` contained incorrect shapes and output names, which caused startup failures. Those configs have been removed so Triton can infer the correct settings directly from the model graphs.

If Triton is run with `strict_model_config=0`, it will automatically create valid configs for these models at startup. If persisted configs are desired, fetch the generated configuration from the Triton HTTP API (`/v2/models/<model>/config`) after startup and save it as `models/<model>/config.pbtxt`.
