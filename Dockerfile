FROM ubuntu:latest
LABEL authors="noab"

RUN apt-get -y update && apt-get install -y --no-install-recommends python3 cmake build-essential openmpi-bin libopenmpi-dev

WORKDIR /app

COPY . .

RUN cmake . && make

