FROM jupyter/datascience-notebook:latest

USER ${NB_UID}

RUN pip install dbfread

USER root

RUN apt-get update && apt-get install -y ttf-mscorefonts-installer 

RUN cd /home
WORKDIR /home

RUN mkdir /home/data
RUN mkdir /home/src
RUN mkdir /home/notebooks
RUN mkdir /home/models


ENV NOTEBOOK_ARGS="--NotebookApp.token='' --NotebookApp.password=''"
COPY requirements.txt .


EXPOSE 8888