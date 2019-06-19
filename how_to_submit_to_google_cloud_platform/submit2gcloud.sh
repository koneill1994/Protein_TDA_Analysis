#!/bin/bash

JOB_NAME=`date "+%Y_%m_%d_%H_%M_%S"`

./google-cloud-sdk/bin/gcloud ml-engine jobs submit training job_$JOB_NAME \
    --module-name=trainer.train \
    --package-path=./trainer \
    --job-dir=gs://dist-predict-test \
    --region=us-central1 \
    --config=cloudml.yaml
