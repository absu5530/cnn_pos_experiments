FROM tensorflow/tensorflow:1.12.0-py3

VOLUME /data
WORKDIR /data

RUN apt-get update && apt-get install -y python3-pip

RUN pip3 install beautifulsoup4 nltk

RUN python3 -m nltk.downloader punkt
RUN python3 -m nltk.downloader tagsets
RUN python3 -m nltk.downloader averaged_perceptron_tagger

ENTRYPOINT ["/bin/bash"]
