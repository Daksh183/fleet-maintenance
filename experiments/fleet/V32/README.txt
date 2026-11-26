Fleet maintenance | v32 build and validation

Submission file in this folder
submission_v32_hist_kernel.csv is the new experimental v32 candidate. The old
v17 file with the recorded 16,826 public score is kept with the historical
project files.

Validation and decision
Across the three development repeats, average source cost per vehicle fell from
0.85796 to 0.84040. Simulated target-depot cost fell from 2.11137 to 2.08690
per vehicle. V32 then increased ordinary holdout cost from 0.76993 to 0.77212
per vehicle, about 24.59 total cost units across 11,200 vehicles. Simulated
target-depot holdout cost fell from 1.90318 to 1.89636 per vehicle, then the
paired difference reversed to +0.01014 after excluding the most influential
vehicle. This is exploratory offline evidence. The team reported a V32 public score of 16,991.5.

The v32 file passed the 20,000-ID, binary-flag, and nonnegative-budget checks.
It flags 1,768 vehicles and applies the new estimate to 7,678 supported test
vehicles. The locally rebuilt historical policy differs from the archived
v17 file on 59 flags. The paired evaluation used a common local tree to
isolate the new estimator.

Files
The paired results CSVs contain fold-aggregated comparisons and cost breakdowns.
The decision JSON files report the frozen development and holdout results.
submission_audit.json records the build comparisons, checks, and candidate hash.
run_v32.py and fleet_runtime contain the implementation and its project inputs.

Rebuild on Windows
From PowerShell inside this folder, with Python 3.11.9 and the dependencies in
../fleet_runtime/requirements.txt installed, run the command below. It assumes the
official mlprojectfleet2.zip is at the path you supply. The run directory
receives the rebuilt candidate and its audit.

python .\run_v32.py build --repo ..\fleet_runtime --zip "C:/path/to/mlprojectfleet2.zip" --cache .\run\cache --out .\run

To repeat validation, first generate the caches with
../fleet_runtime/validation/fit_folds.py using --zip, --cache and --seeds 42 100 101.
Generate the holdout cache with --holdout. Then run the same run_v32.py command
with development followed by holdout as the first argument. Use the same output
directory so the holdout command can check the saved development decision.
