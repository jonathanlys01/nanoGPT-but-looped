#!/bin/bash

REPO_ID="HuggingFaceFW/fineweb-edu"


huggingface-cli download $REPO_ID --local-dir /Brain/public/datasets/$REPO_ID --repo-type dataset --include "sample/10BT/*" 
chmod -R 777 /Brain/public/datasets/$REPO_ID