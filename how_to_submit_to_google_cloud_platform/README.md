### Step 0: Setup your [Google Cloud Storage](https://cloud.google.com/storage/)

### Step 1: Download files in this folder

### Step 2: Initialize
```bash
gcloud init
```

### Update GCP storate in 'submit-gcloud-job.sh' with your google storage location
```bash
   --job-dir=gs://YOUR-GCP-STORAGE 
```

### Step 3: Submit your job (generate job id automatically)
```bash
./submit-gcloud-job.sh
```
----------
### References  
* https://medium.com/@natu.neeraj/training-a-keras-model-on-google-cloud-ml-cb831341c196
* https://github.com/Neeraj-Natu/keras-cloud-test
