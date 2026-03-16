# syntax=docker/dockerfile:1.17-labs
ARG BASE_IMAGE=python:3.12-slim

FROM $BASE_IMAGE AS base

ARG DEBIAN_FRONTEND=noninteractive
ARG PROJECT_PATH
ARG NONROOT_USERNAME=user

# python
ENV PYTHONUNBUFFERED=1 \
    \
    # pip
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    \
    # paths
    # this is where our requirements + virtual environment will live
    VENV_PATH="${PROJECT_PATH}/.venv"

# prepend venv to path
ENV PATH="$VENV_PATH/bin:$PATH"

ARG DEBIAN_FRONTEND=noninteractive
ARG PROJECT_PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

################################################################################

FROM base AS prod-prepare

ARG DEBIAN_FRONTEND=noninteractive
ARG NONROOT_USERNAME

    # Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy \
    # uv
    UV_CACHE_DIR="/home/${NONROOT_USERNAME}/.cache/uv"

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt update \
    && apt install -y --no-install-recommends \
        build-essential

RUN useradd -ms /bin/bash ${NONROOT_USERNAME} --user-group || true
USER ${NONROOT_USERNAME}
WORKDIR ${PROJECT_PATH}

# RUN --mount=type=secret,id=GIT_AUTH_TOKEN,required=true,uid=1000,gid=1000 \
#     gh auth login --with-token < /run/secrets/GIT_AUTH_TOKEN
# RUN gh auth setup-git

# RUN --mount=from=ghcr.io/astral-sh/uv:latest,source=/uv,target=/bin/uv \
#     --mount=from=mikefarah/yq:latest,source=/usr/bin/yq,target=/bin/yq \
#     --mount=type=cache,target=/opt/conda/pkgs \
#     --mount=type=bind,source=environment.yml,target=${PROJECT_PATH}/environment.yml \
    # uv venv --python python3.12 && \
    # yq '.dependencies[] | (select(type=="!!str") // .pip[])' environment.yml \
    #     | sed -E 's/(==|=).*//' \
    #     | grep -v '^_' \
    #     # | grep -vE '^_|^lib|^blas|^cpuonly|^mkl|^ld_impl|^ffmpeg|^icu|^openssl|^python_abi|^tbb|^tk|^lz4-c|^lzma|^xz|^aws' \
    #     | grep -E '^(aiohttp|attrs|anyio|click|colorama|dataclasses-json|datasets|dill|filelock|huggingface_hub|...)$' \
    #     # | tr '\n' ' ' \
    #     | xargs uv pip install --no-cache-dir

# --mount=from=ghcr.io/mamba-org/micromamba:latest,source=/usr/bin/micromamba,target=/bin/mamba \
        # --prefer-binary --find-links https://conda.anaconda.org/conda-forge/win-64/ --find-links https://conda.anaconda.org/conda-forge/noarch/ \
         #   && \
# RUN --mount=from=ghcr.io/mamba-org/micromamba:latest,source=/usr/bin/micromamba,target=/bin/mamba \
#     --mount=type=cache,target=/opt/conda/pkgs \
#     --mount=type=bind,source=environment.yml,target=${PROJECT_PATH}/environment.yml \
#     mamba env create -p ${VENV_PATH} -f ${PROJECT_PATH}/environment.yml --no-pin

    # uv itself
RUN --mount=from=ghcr.io/astral-sh/uv:latest,source=/uv,target=/bin/uv \
    # Project files
    --mount=type=bind,source=pyproject.toml,target=${PROJECT_PATH}/pyproject.toml \
    --mount=type=bind,source=uv.lock,target=${PROJECT_PATH}/uv.lock \
    # If there are projects need ssh access
    # --mount=type=ssh \
    uv sync --frozen --no-install-project --no-install-workspace --no-dev

# install runtime deps - with project itself
COPY --exclude=**/*.py --chown=${NONROOT_USERNAME}:${NONROOT_USERNAME} . ${PROJECT_PATH}
    # uv download cache
    # uv itself
RUN --mount=from=ghcr.io/astral-sh/uv:latest,source=/uv,target=/bin/uv \
    # Project files
    --mount=type=bind,source=pyproject.toml,target=${PROJECT_PATH}/pyproject.toml \
    --mount=type=bind,source=uv.lock,target=${PROJECT_PATH}/uv.lock \
    uv sync --locked --no-dev

# for compatibility with prod stage COPY when uv is using python from /usr/bin
RUN mkdir -p /home/${NONROOT_USERNAME}/.local/share/uv/python

#################################################################################

FROM base AS prod
ARG DEBIAN_FRONTEND=noninteractive
ARG NONROOT_USERNAME

RUN useradd -ms /bin/bash ${NONROOT_USERNAME} --user-group || true
USER ${NONROOT_USERNAME}


WORKDIR ${PROJECT_PATH}

RUN chown ${NONROOT_USERNAME}:${NONROOT_USERNAME} ${PROJECT_PATH}

COPY --chown=${NONROOT_USERNAME}:${NONROOT_USERNAME} --from=prod-prepare /home/${NONROOT_USERNAME}/.local/share/uv/python /home/${NONROOT_USERNAME}/.local/share/uv/python
COPY --chown=${NONROOT_USERNAME}:${NONROOT_USERNAME} --from=prod-prepare ${VENV_PATH} ${VENV_PATH}

USER ${NONROOT_USERNAME}

COPY --exclude=.devcontainer/ --chown=${NONROOT_USERNAME}:${NONROOT_USERNAME} . .

CMD ["python", "main.py"]
