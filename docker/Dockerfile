# syntax=docker/dockerfile:1.7
#
# Reference environment for Àgbẹ̀.
#
# Pinned to ubuntu:22.04 because that is the ADTC Standard Laptop OS. Building
# here rather than on the developer's host means the Gate 2 reproducibility
# audit runs the same userland we developed against, on any machine.
#
# Ubuntu 22.04 ships Python 3.10 as `python3`; we use it rather than adding a
# deadsnakes PPA so the image stays close to a stock target machine.
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # NOT PIP_NO_CACHE_DIR. Downloads are cached via a BuildKit cache mount
    # below, which keeps the cache out of the image while making a retry cost
    # nothing. On a slow link the alternative is re-downloading a 65 MB
    # archive on every failed attempt, which turned each build iteration into
    # a half-hour wait.
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # pip's default read timeout is 15 seconds. On a slow or contended link
    # that is far too short for a 65 MB source archive: a build here died
    # twice with ReadTimeoutError partway through llama-cpp-python. These
    # values let the build survive a connection that is slow rather than
    # broken, which is the common case on the infrastructure this project
    # exists to serve.
    PIP_DEFAULT_TIMEOUT=180 \
    PIP_RETRIES=10

# Transient DNS and mirror failures break long apt runs, and this one pulls a
# full toolchain. A first build here died after 67 packages with
# "Could not resolve 'archive.ubuntu.com'". Retries and longer timeouts make
# the build survive a flaky link, which matters most for the person
# reproducing it on unfamiliar infrastructure.
RUN printf '%s\n' \
        'Acquire::Retries "8";' \
        'Acquire::http::Timeout "30";' \
        'Acquire::https::Timeout "30";' \
        'Acquire::ForceIPv4 "true";' \
    > /etc/apt/apt.conf.d/80-agbe-resilience

# build-essential + cmake are needed to compile llama-cpp-python against
# OpenBLAS. libgomp1 provides the OpenMP runtime llama.cpp threads on.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-dev \
        python3-pip \
        build-essential \
        cmake \
        ninja-build \
        git \
        curl \
        ca-certificates \
        libopenblas-dev \
        libgomp1 \
        # ggml-blas/CMakeLists.txt calls find_package(PkgConfig) when the BLAS
        # backend is enabled. Without it CMake configuration fails *after*
        # successfully locating libopenblas, which makes the error read as a
        # BLAS problem when it is not.
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/local/bin/python \
    && python -m pip install --upgrade pip==24.3.1 setuptools==75.6.0 wheel==0.45.1

WORKDIR /app

# Build llama-cpp-python against OpenBLAS. On a CPU-only target, prompt
# processing is compute-bound and BLAS materially improves prefill latency;
# token generation stays bandwidth-bound and is unaffected.
ENV CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS"
ENV FORCE_CMAKE=1

COPY requirements.txt /app/requirements.txt

# Split deliberately into two layers.
#
# llama-cpp-python is a 65 MB source archive that then compiles for several
# minutes; everything else is small pure-Python or prebuilt wheels. Installed
# together, a network failure on the big one discards the cheap ones too and
# every retry re-downloads all of them. Split, the light layer caches after
# the first success and retries only repeat the expensive download.
RUN --mount=type=cache,target=/root/.cache/pip \
    grep -v '^llama-cpp-python' /app/requirements.txt > /tmp/light.txt \
    && python -m pip install -r /tmp/light.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install -r /app/requirements.txt

# Test tooling last, so adding or bumping it never invalidates the layer above
# - which compiles llama.cpp and takes minutes.
COPY requirements-dev.txt /app/requirements-dev.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install -r /app/requirements-dev.txt

# Fail the build if anything pulled torch in transitively. Cheaper to catch
# here than to discover a 1GB regression during a benchmark run.
RUN if python -c "import torch" 2>/dev/null; then \
        echo "FATAL: torch present in image - see requirements.txt header" >&2; \
        exit 1; \
    fi

COPY . /app

ENV AGBE_MODELS_DIR=/app/models \
    AGBE_CORPUS_DIR=/app/corpus \
    AGBE_INDEX_DIR=/app/index

EXPOSE 8000

CMD ["python", "-m", "agbe.ui.server"]
