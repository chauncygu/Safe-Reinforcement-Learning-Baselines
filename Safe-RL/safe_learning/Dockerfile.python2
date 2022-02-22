FROM continuumio/miniconda:4.5.11

# Install build essentials and clean up
RUN apt-get update --quiet \
  && apt-get install -y --no-install-recommends --quiet build-essential \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Update conda, install packages, and clean up
RUN conda install python=2.7 --yes --quiet \
  && conda clean --yes --all \
  && hash -r

# Copy the main code
COPY . /code
RUN cd /code \
  && pip install pip==18.1 \
  && pip install numpy==1.14.5 \
  && pip install -e .[test] --process-dependency-links \
  && rm -rf /root/.cache

WORKDIR /code
