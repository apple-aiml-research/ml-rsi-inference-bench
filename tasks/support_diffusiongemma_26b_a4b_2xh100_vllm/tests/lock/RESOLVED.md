# Lock provenance

- resolved: 2026-09-29T18:38:40Z inside nvidia/cuda:13.0.3-devel-ubuntu24.04 (x86_64, Python 3.12.3)
- resolver: uv 0.9.28 (pip compile, --generate-hashes, --exclude-newer 2026-06-12T05:16:14Z)
- inputs at vllm 1ce3cdc5c14f656f81f91e264d557e0b6e6fea54:
  - requirements/cuda.txt        sha256 b15cdeba2ca41715bf4376b076d1236ea55034f771af171e4198cffceda97ad7
  - requirements/common.txt      sha256 211b170b3a58d30522b2f4c26733cb0c74cdb04fd6d5cab2c1bcde9af592fc76   (pulled in by cuda.txt's -r)
  - requirements/build/cuda.txt  sha256 f60e5120e13b7e66e10f01293d46692bfd502e1457bafd824c5bcce7822751bd
  - requirements/test/cuda.txt   sha256 b71f874a9f7941ad61b352bdd4ec5d9f23b783f7cc810565b0c87855c902cf8e   (316 pins, each required at its version)
- additions (extra.in): transformers==5.5.3
- indexes: https://pypi.org/simple, https://download.pytorch.org/whl/cu130,
           https://flashinfer.ai/whl/cu130, https://flashinfer.ai/whl (unsafe-best-match)
- result: 402 packages in requirements.lock
- FlashInfer JIT cache: flashinfer-jit-cache==0.6.12+cu130 from https://flashinfer.ai/whl/cu130 (--no-deps)
- tools: huggingface-hub==1.10.2, hf_transfer
- check 1: 316 test pins; 0 not honoured: []
- check 2: 86 packages not pinned by the test lock; 0 first uploaded after the base date: []
- resolver warnings: 1371 index files without an upload date, for cuda_bindings cuda_pathfinder cuda_toolkit filelock flashinfer_cubin flashinfer_python fsspec mpmath networkx numpy nvidia_cublas nvidia_cuda_cccl nvidia_cuda_cupti nvidia_cuda_nvrtc nvidia_cuda_runtime nvidia_cudnn_cu13 nvidia_cufft nvidia_cufile nvidia_curand nvidia_cusolver nvidia_cusparse nvidia_cusparselt_cu13 nvidia_nccl_cu13 nvidia_nvjitlink nvidia_nvshmem_cu13 nvidia_nvtx pillow sympy typing_extensions 
  (every such package is covered by check 1 or check 2)

## Check 2, per package

    anthropic==0.109.1 first upload 2026-06-09T23:55:23.348791Z ok
    apache-tvm-ffi==0.1.9 first upload 2026-02-27T19:27:03.001454Z ok
    astor==0.8.1 first upload 2019-12-10T01:50:33.628982Z ok
    blake3==1.0.8 first upload 2025-10-14T06:45:09.621234Z ok
    build==1.5.0 first upload 2026-04-30T03:18:23.644906Z ok
    cbor2==6.1.2 first upload 2026-06-02T19:00:21.183771Z ok
    cloudpickle==3.1.2 first upload 2025-11-03T09:25:25.534602Z ok
    cmake==4.3.2 first upload 2026-04-23T21:50:16.764674Z ok
    compressed-tensors==0.17.0 first upload 2026-06-03T16:49:15.672396Z ok
    cuda-python==13.0.3 first upload 2025-10-21T15:48:59.222618Z ok
    cuda-tile==1.3.0 first upload 2026-04-20T15:51:01.914372Z ok
    depyf==0.20.0 first upload 2025-10-13T12:33:33.647570Z ok
    diskcache==5.6.3 first upload 2023-08-31T06:11:58.822453Z ok
    distro==1.9.0 first upload 2023-12-24T09:54:30.421191Z ok
    docstring-parser==0.18.0 first upload 2026-04-14T04:09:18.638975Z ok
    fastapi-cli==0.0.21 first upload 2026-02-11T15:27:58.181542Z ok
    fastapi-cloud-cli==0.11.0 first upload 2026-01-15T09:51:33.681494Z ok
    fastar==0.11.0 first upload 2026-04-13T17:07:52.788957Z ok
    flashinfer-cubin==0.6.12 first upload 2026-05-29T23:45:27.579138Z ok
    flashinfer-python==0.6.12 first upload 2026-05-29T23:45:13.477313Z ok
    gguf==0.19.0 first upload 2026-05-06T13:04:02.588361Z ok
    httptools==0.8.0 first upload 2026-05-25T22:16:50.434730Z ok
    httpx-sse==0.4.3 first upload 2025-10-10T21:48:21.158886Z ok
    humming-kernels==0.1.4 first upload 2026-06-04T03:24:01.897660Z ok
    ijson==3.5.0 first upload 2026-02-24T03:56:40.591647Z ok
    interegular==0.3.3 first upload 2024-01-06T23:01:20.829985Z ok
    jiter==0.15.0 first upload 2026-05-19T10:06:51.944679Z ok
    lark==1.2.2 first upload 2024-08-13T19:48:58.603221Z ok
    llguidance==1.7.6 first upload 2026-06-03T20:12:55.042931Z ok
    lm-format-enforcer==0.11.3 first upload 2025-08-24T19:37:46.325940Z ok
    loguru==0.7.3 first upload 2024-12-06T11:20:54.538869Z ok
    mcp==1.27.2 first upload 2026-05-29T17:16:02.442669Z ok
    ml-dtypes==0.5.4 first upload 2025-11-17T22:31:31.367673Z ok
    model-hosting-container-standards==0.1.15 first upload 2026-05-05T18:22:27.819372Z ok
    msgspec==0.21.1 first upload 2026-04-12T21:43:45.062918Z ok
    ninja==1.13.0 first upload 2025-08-11T15:09:50.971151Z ok
    nvidia-cuda-cccl==13.3.3.3.1 first upload 2026-05-26T16:27:45.179309Z ok
    nvidia-cuda-crt==13.3.33 first upload 2026-05-26T16:28:40.446251Z ok
    nvidia-cuda-nvcc==13.2.78 first upload 2026-04-13T09:42:17.478505Z ok
    nvidia-cuda-tileiras==13.2.78 first upload 2026-04-13T09:48:30.107084Z ok
    nvidia-cudnn-frontend==1.25.0 first upload 2026-06-10T21:04:33.452886Z ok
    nvidia-cutlass-dsl==4.5.2 first upload 2026-05-25T03:38:56.364140Z ok
    nvidia-cutlass-dsl-libs-base==4.5.2 first upload 2026-05-25T03:46:33.526814Z ok
    nvidia-cutlass-dsl-libs-cu13==4.5.2 first upload 2026-05-25T03:39:21.998370Z ok
    nvidia-ml-py==13.610.43 first upload 2026-06-01T18:54:07.704099Z ok
    nvidia-nvvm==13.2.78 first upload 2026-04-13T10:02:00.345543Z ok
    openai==2.41.1 first upload 2026-06-10T16:10:35.756209Z ok
    opentelemetry-exporter-otlp==1.35.0 first upload 2025-07-11T12:23:10.679681Z ok
    opentelemetry-exporter-otlp-proto-common==1.35.0 first upload 2025-07-11T12:23:11.713418Z ok
    opentelemetry-exporter-otlp-proto-grpc==1.35.0 first upload 2025-07-11T12:23:12.957175Z ok
    opentelemetry-exporter-otlp-proto-http==1.35.0 first upload 2025-07-11T12:23:13.906429Z ok
    opentelemetry-semantic-conventions-ai==0.4.13 first upload 2025-08-22T10:14:16.477377Z ok
    outlines-core==0.2.14 first upload 2026-01-09T15:58:07.323443Z ok
    partial-json-parser==0.2.1.1.post7 first upload 2025-11-17T07:27:40.457273Z ok
    prometheus-fastapi-instrumentator==7.1.0 first upload 2025-03-19T19:35:04.323771Z ok
    py-cpuinfo==9.0.0 first upload 2022-10-25T20:38:06.303382Z ok
    pybase64==1.4.3 first upload 2025-12-06T13:22:27.396884Z ok
    pydantic-settings==2.14.1 first upload 2026-05-08T13:40:04.958540Z ok
    pyelftools==0.33 first upload 2026-05-29T12:56:20.587302Z ok
    pyproject-hooks==1.2.0 first upload 2024-09-29T09:24:11.978642Z ok
    python-dotenv==1.2.2 first upload 2026-03-01T16:00:25.090725Z ok
    python-json-logger==4.1.0 first upload 2026-03-29T04:39:55.266869Z ok
    python-multipart==0.0.32 first upload 2026-06-04T16:18:57.319904Z ok
    pyzmq==27.1.0 first upload 2025-09-08T23:07:26.274469Z ok
    quack-kernels==0.5.0 first upload 2026-05-29T05:00:24.679039Z ok
    rich-toolkit==0.20.1 first upload 2026-06-05T08:56:57.679583Z ok
    rignore==0.7.6 first upload 2025-11-05T20:40:48.940469Z ok
    semantic-version==2.10.0 first upload 2022-05-26T13:35:21.206388Z ok
    sentencepiece==0.2.1 first upload 2025-08-12T06:58:59.807765Z ok
    sentry-sdk==2.62.0 first upload 2026-06-08T13:23:47.486410Z ok
    setproctitle==1.3.7 first upload 2025-09-05T12:49:07.732772Z ok
    setuptools-rust==1.12.1 first upload 2026-03-26T08:02:10.017393Z ok
    setuptools-scm==10.0.5 first upload 2026-03-27T15:57:03.969596Z ok
    sse-starlette==2.1.3 first upload 2024-08-01T08:52:48.659065Z ok
    supervisor==4.3.0 first upload 2025-08-23T18:25:00.767179Z ok
    tilelang==0.1.9 first upload 2026-04-22T09:14:39.366735Z ok
    tokenspeed-mla==0.1.2 first upload 2026-05-13T03:30:54.486286Z ok
    tokenspeed-triton==3.7.10.post20260531 first upload 2026-05-31T01:28:48.605196Z ok
    torch-c-dlpack-ext==0.1.5 first upload 2026-01-12T11:24:31.817862Z ok
    uvloop==0.22.1 first upload 2025-10-16T22:16:11.430849Z ok
    vcs-versioning==1.1.1 first upload 2026-03-27T20:42:40.450717Z ok
    watchfiles==1.2.0 first upload 2026-05-18T04:30:05.440008Z ok
    websockets==16.0 first upload 2026-01-10T09:22:21.280709Z ok
    wheel==0.47.0 first upload 2026-04-22T15:51:26.296257Z ok
    xgrammar==0.2.2 first upload 2026-06-11T19:01:11.995491Z ok
    z3-solver==4.15.4.0 first upload 2025-10-29T18:11:46.233386Z ok
    86 packages not pinned by the test lock; 0 first uploaded after the base date: []
