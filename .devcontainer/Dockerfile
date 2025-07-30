FROM mcr.microsoft.com/devcontainers/python:1-3.11-bookworm AS d3lta-prod

ENV PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    \
    # Poetry
    # https://python-poetry.org/docs/configuration/#using-environment-variables
    POETRY_VERSION=2.1.1 \
    # make poetry install to this location
    POETRY_HOME="/opt/poetry" \
    # do not ask any interactive questions
    POETRY_NO_INTERACTION=1 \
    # never create virtual environments automatically
    POETRY_VIRTUALENVS_CREATE=false

RUN pip install --no-cache-dir --upgrade pip
RUN pipx install poetry==${POETRY_VERSION}

WORKDIR /app
COPY pyproject.toml poetry.lock setup.py README.md ./
# pre-install dependencies
RUN --mount=type=cache,target=/root/.cache poetry install --no-root

COPY notebooks /app/notebooks/
COPY d3lta /app/d3lta/
RUN --mount=type=cache,target=/root/.cache poetry install

FROM d3lta-prod AS d3lta-dev

RUN --mount=type=cache,target=/root/.cache poetry install --with dev

# install nektos/act as specified in https://nektosact.com/installation/index.html#bash-script
# the -b flag specifies the target directory (cf. https://github.com/nektos/act/blob/61396d8085a9d812cebf94fa954f5938d48bf2b9/install.sh#L13)
RUN curl --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash -s -- -b /usr/bin
