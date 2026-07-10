FROM postgres:18

RUN apt-get update && apt-get install -y \
    git build-essential postgresql-server-dev-18 cmake libicu-dev curl libkrb5-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/pgvector/pgvector.git \
    && cd pgvector && make OPTFLAGS="" && make install && cd .. && rm -rf pgvector

RUN git clone --depth 1 --branch v1.1.0 https://github.com/timescale/pg_textsearch.git \
    && cd pg_textsearch && make && make install && cd .. && rm -rf pg_textsearch

RUN git clone --depth 1 https://github.com/dddanielliu/pg_jieba.git \
    && cd pg_jieba && git submodule update --init --recursive \
    && mkdir build && cd build \
    && cmake .. && make CFLAGS="-O2" CXXFLAGS="-O2" -j$(nproc) && make install \
    && cd ../.. && rm -rf pg_jieba

RUN echo "shared_preload_libraries = 'pg_textsearch, pg_jieba'" >> /usr/share/postgresql/postgresql.conf.sample
