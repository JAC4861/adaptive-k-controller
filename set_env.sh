#!/usr/bin/env bash

export PROJECT_HOME=$(pwd)
export LOG_DIR="$PROJECT_HOME/logs"
export DATAPATH="$PROJECT_HOME/data"
export PYTHONPATH="$PROJECT_HOME:$PYTHONPATH"

conda activate hgcn_controller
