FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# install utilities
RUN apt-get update && \
    apt-get install --no-install-recommends -y curl

# Installing python dependencies
RUN apt-get install -y python3
RUN python3 -m pip --no-cache-dir install --upgrade pip && \
    python3 --version && \
    pip3 --version

WORKDIR /code

# Copy app files
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY ./app /code/app
COPY ./regression /code/regression
# Copy pickle files
COPY ./pickle_dump /code/pickle_dump
# Copy model files
COPY ./model /code/model
COPY ./run.sh /run.sh
RUN chmod +x /run.sh

# Installing python modules

EXPOSE 8000

ARG MODE
ENV SCRIPT_MODE $MODE
CMD ["/run.sh"]

# docker build --build-arg MODE="regression" -t test5 .
# docker run -ti --gpus all --name test5 -p 8000:8000 test5
# docker run -ti --name test5 -p 8000:8000 test5