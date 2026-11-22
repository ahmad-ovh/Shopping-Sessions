FROM debian:trixie-slim AS build
RUN apt-get update \
 && apt-get install -y --no-install-recommends g++ libc6-dev \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY src/ ./src/
RUN g++ -std=c++20 -O3 -march=x86-64-v3 -mtune=cascadelake \
        -fno-exceptions -fno-rtti -fno-plt -pthread \
        -Wall -Wextra -Wno-unused-parameter \
        -c src/entry_avx2.cpp -o /tmp/entry_avx2.o \
 && g++ -std=c++20 -O3 -march=cascadelake -Wa,-mbranches-within-32B-boundaries \
        -fno-exceptions -fno-rtti -fno-plt -pthread \
        -Wall -Wextra -Wno-unused-parameter \
        -c src/entry_avx512.cpp -o /tmp/entry_avx512.o \
 && g++ -std=c++20 -O3 -march=x86-64-v2 \
        -fno-exceptions -fno-rtti -fno-plt -pthread \
        -Wall -Wextra -Wno-unused-parameter \
        -c src/main.cpp -o /tmp/main.o \
 && g++ -static -pthread -O3 \
        /tmp/main.o /tmp/entry_avx2.o /tmp/entry_avx512.o -o /hyperion \
 && strip /hyperion

FROM scratch
COPY --from=build /hyperion /hyperion
USER 1000:1000
ENTRYPOINT ["/hyperion"]
