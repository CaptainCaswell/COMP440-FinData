#!/bin/bash

CLASSIFY_JOB=$(sbatch --parsable classify.sbatch)

echo "Classification array submitted: $CLASSIFY_JOB"

sbatch --dependency=afterok:$CLASSIFY_JOB classify_combine.sbatch