ARG POSTGRES_BASE_IMAGE=docker.io/postgres:18.3-bookworm
ARG PGVECTORSCALE_VERSION=0.9.0

FROM ${POSTGRES_BASE_IMAGE} AS builder
ARG PGVECTORSCALE_VERSION

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    clang \
    curl \
    git \
    build-essential \
    llvm-dev \
    libssl-dev \
    pkg-config \
    postgresql-server-dev-18 \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

RUN install -d /usr/share/postgresql-common/pgdg \
    && curl -o /usr/share/postgresql-common/pgdg/apt.postgresql.org.asc --fail https://www.postgresql.org/media/keys/ACCC4CF8.asc \
    && echo "deb [signed-by=/usr/share/postgresql-common/pgdg/apt.postgresql.org.asc] https://apt.postgresql.org/pub/repos/apt bookworm-pgdg main" > /etc/apt/sources.list.d/pgdg.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends postgresql-18-pgvector \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /tmp
RUN git clone --depth 1 --branch ${PGVECTORSCALE_VERSION} https://github.com/timescale/pgvectorscale.git
WORKDIR /tmp/pgvectorscale/pgvectorscale
RUN cargo install --locked cargo-pgrx --version 0.16.1
RUN cargo pgrx init --pg18 /usr/lib/postgresql/18/bin/pg_config
RUN cargo pgrx install --release

FROM ${POSTGRES_BASE_IMAGE}

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

RUN install -d /usr/share/postgresql-common/pgdg \
    && curl -o /usr/share/postgresql-common/pgdg/apt.postgresql.org.asc --fail https://www.postgresql.org/media/keys/ACCC4CF8.asc \
    && echo "deb [signed-by=/usr/share/postgresql-common/pgdg/apt.postgresql.org.asc] https://apt.postgresql.org/pub/repos/apt bookworm-pgdg main" > /etc/apt/sources.list.d/pgdg.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends postgresql-18-pgvector \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/lib/postgresql/18/lib/*vectorscale* /usr/lib/postgresql/18/lib/
COPY --from=builder /usr/share/postgresql/18/extension/*vectorscale* /usr/share/postgresql/18/extension/
