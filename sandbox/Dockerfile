# generalreasoning/dsbc_tools — self-contained agent-sandbox image for
# GeneralReasoning/DSBC.
#
# The DSBC agent works via the CLIToolset (writes + runs Python / shell to
# analyze the datasets mounted read-only at /workspace), so its sandbox needs a
# full data-science Python stack + common CLIs. This image is intentionally
# self-contained (does NOT derive from generalreasoning/python-ds) so DSBC owns
# its exact sandbox environment.
#
# Published to Docker Hub as generalreasoning/dsbc_tools by
# .github/workflows/sandbox-image.yml on any change to this Dockerfile, tagged
# :latest and :<git-sha>. dsbc.py pins an immutable release tag:
#   SandboxSettings(image="generalreasoning/dsbc_tools:v1.0.0", ...)
FROM python:3.13

# --- ds layer: build toolchain + core data-science stack ---
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        numpy \
        pandas \
        matplotlib \
        scikit-learn \
        seaborn \
        jupyter \
        openpyxl

# --- tools layer: common CLIs an analysis agent may reach for ---
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        wget \
        git \
        file \
        unzip \
        gzip \
        jq \
        vim-common \
    && rm -rf /var/lib/apt/lists/*

CMD ["python3"]
