A dashboard for rating unit activity drift (potentially due to over-splitting units) based on
raster plots across task trials.

## Run
git clone then `uvx run panel app.py --autoreload`

## Additional requirements
S3 credentials in a `.env` file or in AWS config files (see [boto3 docs](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html)).

## Virtual machine
Put `app.py` in folder of dashboard apps, then run with: 
`panel serve src/dashboard_test/apps/*.py --autoreload --address dr-dashboard.corp.alleninstitute.org --port 9000 --admin --allow-websocket-origin=* --keep-alive 1000` 