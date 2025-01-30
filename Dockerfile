FROM ubuntu:latest
LABEL authors="noab"

RUN apt-get -y update && apt-get install -y --no-install-recommends python3
RUN apt-get -y update && apt-get install -y --no-install-recommends cmake
RUN apt-get -y update && apt-get install -y --no-install-recommends build-essential

WORKDIR /app

COPY . .

RUN cmake .
RUN make

