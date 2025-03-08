FROM python:3.9

RUN apt-get update && \
    apt-get install -y \
        apt-utils \
        autoconf \
        automake \
        bzip2 \
        ca-certificates \
        cmake \
        g++ \
        git \
        htop \
        ipython3 \
        less \
        libtool \
        locales \
        make \
        nano \
        pkg-config \
        python3-dev python3-launchpadlib \
        rsync \
        software-properties-common \
        time \
        tmux \
        unzip \
        wget \
        xz-utils \
        zlib1g-dev

# Audio specific
RUN apt-get install -y \
        sox \
        libsox-fmt-mp3 \
        libsox-dev \
        ffmpeg \
        libssl-dev \
        libsndfile1 \
        portaudio19-dev \
        libcurl4-openssl-dev \
        xvfb

# Web-Crawling specific
RUN add-apt-repository ppa:mozillateam/ppa && apt-get update && apt-get install -y firefox-esr

# Python
RUN python3 -m pip install --upgrade pip
COPY requirements.txt ./
RUN pip3 install numpy numpy typing_extensions
RUN pip3 install -r requirements.txt
RUN rm requirements.txt

# Speechbrain
RUN git clone https://github.com/speechbrain/speechbrain /opt/speechbrain \
    && cd /opt/speechbrain/ \
    && pip3 install -r requirements.txt \
    && SETUPTOOLS_USE_DISTUTILS=stdlib pip3 install -e .

# Whisper
RUN pip3 install git+https://github.com/openai/whisper.git 

# Tweaks (for the SKIPPING trick (look for 'SKIPPING' in the code), useful to restart training experiments with transformers trainer, when the jobs stopped in the middle of a huge epoch)
# COPY docker/transformers_modified/trainer.py /usr/local/lib/python3.9/site-packages/transformers/

# Fix for hyperpyyaml
RUN pip3 install git+https://github.com/speechbrain/HyperPyYAML@1e47fa63982933cd7fb01e1e6e063be492fddeab

# Used to scrap
COPY tools/requirements.txt ./
RUN pip3 install -r requirements.txt
RUN rm requirements.txt

# Locale
# RUN sed -i '/fr_FR.UTF-8/s/^# //g' /etc/locale.gen && locale-gen
# ENV LANG fr_FR.UTF-8  
# ENV LANGUAGE fr_FR:en  
# ENV LC_ALL fr_FR.UTF-8

# ENV LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

ENV WANDB_DISABLED=1

ENTRYPOINT ["/bin/bash"]