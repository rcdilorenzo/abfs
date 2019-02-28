FROM ubuntu:16.04

ARG WEIGHTS_S3
ARG MODEL_S3
ARG MAPBOX_API_KEY

ENV PORT 80
ENV WEIGHTS_S3 WEIGHTS_S3
ENV MODEL_S3 MODEL_S3
ENV MAPBOX_API_KEY MAPBOX_API_KEY

RUN apt-get update
RUN apt-get install -y software-properties-common git curl gcc make zlib1g-dev libssl-dev libreadline-dev libbz2-dev libsqlite0-dev

# OpenCV libraries
RUN apt-get install -y libsm6 libxext6 libxrender1 libfontconfig1 libice6

RUN curl https://pyenv.run | bash
ENV PATH /root/.pyenv/shims:/root/.pyenv/bin:$PATH

COPY . /app
WORKDIR /app

RUN pyenv install 3.6.8
RUN pyenv virtualenv 3.6.8 abfs-env
RUN pyenv rehash

RUN add-apt-repository -y ppa:ubuntugis/ubuntugis-unstable
RUN apt-get update

RUN pip install numpy

RUN apt-get install -y aptitude
RUN aptitude install -y libgdal-dev

RUN pip install --global-option=build_ext --global-option="-I/usr/include/gdal" GDAL==`gdal-config --version`

RUN pip install -r requirements.txt

EXPOSE 80

CMD ["./run", "PORT=80"]
