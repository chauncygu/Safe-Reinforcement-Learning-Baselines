FROM continuumio/miniconda3

# Install build essentials and clean up
RUN apt-get update --quiet \
  && apt-get install -y --no-install-recommends --quiet build-essential \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Update conda, install packages, and clean up
RUN conda update conda --yes --quiet \
  && conda install python=3.5 pip numpy scipy pandas --yes --quiet \
  && conda clean --yes --all \
  && hash -r

# Get the requirements files (seperate from the main body)
COPY requirements.txt requirements_dev.txt /reqs/

# Install requirements and clean up
RUN pip --no-cache-dir install -r /reqs/requirements.txt \
  && pip --no-cache-dir install -r /reqs/requirements_dev.txt \
  && pip install jupyter jupyterlab dumb-init \
  && rm -rf /root/.cache \
  && rm -rf /reqs

# Manually install GPflow and clean up
RUN git clone --depth=1 --branch=0.4.0 https://github.com/GPflow/GPflow.git \
  && cd GPflow \
  && python setup.py install \
  && rm -rf /GPflow

# Output scrubber for jupyter
ADD scripts/jupyter_output.py /

RUN jupyter notebook --generate-config \
  && cat /jupyter_output.py >> /root/.jupyter/jupyter_notebook_config.py \
  && rm /jupyter_output.py

WORKDIR /code

# Make sure Ctrl+C commands can be forwarded
ENTRYPOINT ["dumb-init", "--"]

CMD python setup.py develop \
  && jupyter lab --ip="0.0.0.0" --no-browser --allow-root
