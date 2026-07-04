FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        cargo \
        clang \
        cmake \
        curl \
        freeglut3-dev \
        g++ \
        git \
        git-lfs \
        ladspa-sdk \
        libasound2-dev \
        libcurl4-openssl-dev \
        libfontconfig1-dev \
        libfreetype6-dev \
        libgl1-mesa-dev \
        libglu1-mesa-dev \
        libgtk-3-dev \
        libjack-jackd2-dev \
        libwebkit2gtk-4.1-dev \
        libx11-dev \
        libxcomposite-dev \
        libxcursor-dev \
        libxinerama-dev \
        libxrandr-dev \
        mesa-common-dev \
        ninja-build \
        pkg-config \
        python3 \
        python3-pip \
        python3-venv \
        rustc \
        xvfb \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . .

RUN git lfs install --system \
    && if [ -d .git ]; then git submodule update --init --recursive; else test -f third_party/ncw/Cargo.toml; fi

RUN python3 -m pip install --break-system-packages poetry \
    && cd python \
    && poetry install --with train,wandb --no-interaction

RUN cmake -B build-fit -DCMAKE_BUILD_TYPE=Release -DBUILD_FIT_TOOLS=ON \
    && cmake --build build-fit --target PianoFit --config Release -j 2

CMD ["piano-fit", "--help"]
