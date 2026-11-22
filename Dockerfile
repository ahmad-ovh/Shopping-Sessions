FROM debian:bookworm-slim AS build
RUN apt-get update \
 && apt-get install -y --no-install-recommends g++ libc6-dev \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY src/ ./src/
RUN g++ -std=c++20 -O3 -march=x86-64-v3 -mtune=cascadelake \
        -flto -fno-exceptions -fno-rtti -fno-plt -static -pthread \
        -Wall -Wextra -Wno-unused-parameter \
        src/main.cpp -o /hyperion \
 && strip /hyperion

FROM scratch
COPY --from=build /hyperion /hyperion
USER 1000:1000
ENTRYPOINT ["/hyperion"]
