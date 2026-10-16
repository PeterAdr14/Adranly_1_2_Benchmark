# OCSR benchmark for Slurm clusters: BioMiner (MolGlyph) and MolNexTR

Reproducible benchmark of optical chemical structure recognition (image to SMILES) tools on public datasets,
written for Northeastern's **Explorer** cluster. Pipeline: `datasets -> inference (resumable) -> scoring -> results/`.

| | |
|---|---|
| Tools | **MolGlyph**, the open-weight model inside BioMiner (main environment, Python 3.10), and **MolNexTR** (own Python 3.8 environment, run in a worker process). OSRA is implemented but disabled. |
| Datasets | **CLEF, UOB, USPTO, JPO** (the benchmarks named in the setup proposal; hosted in MolNexTR's public Hugging Face dataset repo as `real.zip`) and **BioVista**'s OCSR set (BioMiner's benchmark). |
| Metrics | exact match, exact match without stereo, InChIKey match, Tanimoto, valid-SMILES rate, bootstrap 95% CI, runtime/throughput, GPU memory, failure type per image, paired tool comparison with CI. |
| Outputs | `predictions.csv`, `comparison.csv`, `pairwise_comparison.csv`, `failure_summary.csv`, `metrics.json`, `summary.md`, `run_metadata.json`, `requirements.lock.txt`, `figures/*.png`. |

## Getting access and logging in

1. **Make sure your account can use the cluster.** If you only have access through the class, the instructor must have requested classroom access
   (`rc-docs.northeastern.edu`, "Courses on Explorer"). Once granted you get a folder under `/courses/<COURSE>.<TERM>/students/<username>` and can use the
   `courses` (CPU) and `courses-gpu` partitions. A persistent account is a separate request: <https://rc.northeastern.edu/getting-access/>. Questions: rchelp@northeastern.edu.
2. **Connect.** From a terminal: `ssh <username>@login.explorer.northeastern.edu` (Northeastern username and password). Windows: MobaXterm; or use the web portal
   <https://ood.explorer.northeastern.edu> (Open OnDemand; the "Courses" tab offers JupyterLab on `courses` or `courses-gpu`).
   The documentation pages I read do not mention a VPN or two-factor step; if the connection times out off campus, ask Research Computing.
3. **Check what you may use** (on the login node; these are light commands):

   ```bash
   hostname; whoami; echo $HOME
   ls /courses/                                              # your course folder(s)
   sinfo -p courses-gpu --Format=nodes,cpus,gres,statecompact   # GPU types and node availability
   squeue -u $USER
   srun --time=0:30:00 --partition=courses-gpu --gres=gpu:1 --pty /bin/bash   # then: nvidia-smi ; exit
   ```
4. **Copy the project over** (from your own computer): `scp ocsr-benchmark-hpc.zip <username>@login.explorer.northeastern.edu:~/`, then `unzip` it on the cluster.
   (Larger transfers: see the "Transfer Data" and Globus pages of the Explorer documentation.)
5. **Tell the scripts where to work.** Course account: `export OCSR_COURSE_DIR=/courses/<COURSE>.<TERM>/students/$USER` (put it in `~/.bashrc` to keep it).
   Research account: use the `/scratch/$USER` default and `export OCSR_CPU_PARTITION=short OCSR_GPU_PARTITION=gpu OCSR_GPU_MEM=32G`.

## Quick start on Explorer

Run these on a login node (they only submit jobs and edit files; heavy work happens in the jobs).

```bash
# 1. Get the code and put your Hugging Face READ token in a file outside the repository
git clone <this repository> ocsr-benchmark-hpc && cd ocsr-benchmark-hpc
install -m 600 /dev/null ~/.hf_token && echo 'hf_xxx...' > ~/.hf_token     # your token
#    MolGlyph is gated: accept its terms at https://huggingface.co/jiaxianustc/MolGlyph with the same account.

# 2. Check module names, partition limits (and GPU types) once
module avail anaconda3 cuda
scontrol show partition gpu | grep -i -E "MaxTime|DefaultTime"

# 3. Submit everything as a dependency chain: setup -> prefetch -> smoke test -> full benchmark
bash slurm/submit_all.sh                 # or: --skip-setup / --skip-smoke
squeue -u $USER                          # follow progress; logs are in logs/
tail -f logs/*.out
```

Step by step instead (each is an `sbatch` from the repository root):

| Step | Script | Partition | What it does |
|---|---|---|---|
| 1 | `slurm/setup_envs.sbatch` | `courses` (CPU) | Builds two conda environments (Python 3.10 with torch 2.6 for MolGlyph; Python 3.8 with torch 2.4.1 for MolNexTR) and installs the pinned requirements. MolNexTR's code needs a few packages its `requirements.txt` omits (`pystow` was the first): the job installs it and then repairs further missing modules one at a time (up to 8) before its final import check. |
| 2 | `slurm/prefetch.sbatch` | `courses` (CPU) | Downloads datasets, both checkpoints and tool code, then **loads each model once on the CPU** (so import or checkpoint problems show up here, not in the GPU queue); writes `dataset_overview.csv`. |
| 3 | `slurm/smoke.sbatch` | `courses-gpu`, 1 GPU | 20 images per dataset. **Read `throughput_img_per_s` here** to estimate the full run (hours per tool is roughly images / throughput / 3600). |
| 4 | `slurm/benchmark.sbatch` | `courses-gpu`, 1 GPU | The full run. Saves progress after every batch and resubmits itself before the time limit. |

The partitions, memory and time limit come from `slurm/settings.sh` (course-account defaults: `courses`, `courses-gpu`, 12 GB, 8 h); `submit_all.sh` passes them to
`sbatch`. **Do not run a bare `sbatch slurm/<job>.sbatch`**: it uses the partition written in the script (the course default) and fails for accounts without course access.
To submit a single job with your settings use the wrapper: `bash slurm/submit.sh benchmark` (also `setup_envs`, `prefetch`, `smoke`; extra sbatch options go after the name, e.g. `--exclude=d4054`).

Pass options to any job with `sbatch --export=ALL,RUN_ARGS="--set run.max_samples=100 --set run.name=small" slurm/benchmark.sbatch`.
To debug interactively on a GPU (from the cluster docs):

```bash
srun --partition=courses-gpu --nodes=1 --ntasks=1 --gres=gpu:1 --mem=12G --time=01:00:00 --pty /bin/bash   # research accounts: --partition=gpu-interactive
source slurm/env.sh && $OCSR_PY -m ocsr_bench run --set run.name=debug --set run.max_samples=10
```

## Where things live, and why

| Item | Default location | Reason (from the Explorer documentation) |
|---|---|---|
| Datasets, caches, cloned tool code, conda environments | `$OCSR_COURSE_DIR/ocsr_bench` if set, else `/scratch/$USER/ocsr_bench` | Home has a small fixed quota. `/scratch` is large but **purged monthly**, so everything there must be re-creatable. A course folder under `/courses/` is deleted after the course ends (student folders after grades are submitted, per the course guide), so **copy your results out before then**. For long-lived environments use your group's `/projects` folder (`export OCSR_ENV_MAIN=/projects/<group>/...`). |
| Results (`results/`, small) | `$HOME/ocsr_results` | Survives a scratch purge. Copy them to your computer (`scp -r`) or `/projects` before a course account expires. |
| Caches (`HF_HOME`, pip, uv, torch, conda) | under the work directory | Package caches would otherwise fill `$HOME`. |
| Hugging Face token | `~/.hf_token` (`chmod 600`) | Read by `slurm/env.sh`; never written to any output (the audit checks this). |

Job design: package installs and downloads run in **CPU jobs** (`courses`, or `short` for research accounts) (CPU-heavy work does not belong on login nodes, and idle GPUs are
monitored), the **GPU jobs only run inference**, and every GPU request is one GPU (`--gres=gpu:1`). On `courses-gpu` the limits are: **one running job at a time, 12 GB RAM, 8 h maximum, and a job idle for an hour is terminated**
(from the Course Guide); the scripts respect all four (the self-resubmitted job just waits for the previous one to end). All paths and module names can be
overridden by exporting variables before submitting (see the top of `slurm/env.sh`).

## Configuration

`configs/default.yaml` holds all settings. Paths use `${OCSR_WORK_DIR}`-style variables (set by `slurm/env.sh`; `${VAR:-fallback}` is supported),
and any value can be overridden: `python -m ocsr_bench run --set run.max_samples=100 --set datasets.clef.params.max_samples=50`.
`--only-datasets` / `--only-tools` restrict a run, and `"enabled": false` switches an entry off. Default caps are 500 images per proposal dataset
and 1,000 for BioVista (seeded random subsets); set `max_samples: null` for the full sets once the smoke test has given you the throughput.

Commands: `python -m ocsr_bench {selfcheck,preflight,prefetch,overview,run,report}` (`--log-level` goes before the command).
`report` rebuilds figures and the requirements audit from a finished run's files.

## Reproducibility and resuming

* Raw tool outputs are appended to `raw/<tool>__<dataset>.jsonl` after every batch; rerunning the same command skips finished images.
  A run folder is tied to a configuration hash (location-only settings such as `data_dir` are ignored); a changed configuration under the
  same `run.name` is refused, so use a new name.
* Seeds control subsampling and bootstrap resampling; the BioMiner and MolNexTR commits and the package versions of both environments are saved in
  `run_metadata.json`; pin the commits with `biominer_ref` / `repo_ref`.
* Each dataset's sample list is cached in `<work dir>/data/_sample_cache/` the first time it is built (the prefetch job does this on a CPU node), so the GPU job does not walk
  the archive's tens of thousands of files again. Delete that folder, or set `OCSR_NO_SAMPLE_CACHE=1`, to force a rescan.
* A batch that fails is retried image by image, so one bad image cannot stop a long run; the failure is recorded per image.

## Tests

```bash
python -m unittest discover -s tests -t .      # 86 tests, offline, no GPU needed
```

They use stand-in models (including a real worker subprocess), a fake Hugging Face download and a stub chemistry backend, and cover config and
environment-variable handling, dataset layouts, metrics, the full pipeline, resume, the CLI commands and static checks of the Slurm scripts.
RDKit and Open Babel tests are skipped automatically when those packages are missing.

## Reference run and results

The pipeline was run end to end on Explorer: a smoke test (20 images per dataset, one T4) and the full run below (one H200, about 1 hour for both tools). Exact match, with 95% bootstrap CIs
in `results/phase1_benchmark/summary.md` (copy of the run folder; it also holds `comparison.csv`, `pairwise_comparison.csv`, `failure_summary.csv`, `predictions.csv`, `metrics.json`,
`run_metadata.json`, `requirements.lock.txt`, the raw per-image tool outputs in `raw/`, and `figures/`).

| Dataset (images scored) | MolGlyph (BioMiner) | MolNexTR | Paired difference (MolGlyph minus MolNexTR) |
|---|---|---|---|
| CLEF (454) | 0.780 | 0.764 | +0.015 [-0.033, 0.062] (not significant) |
| UOB (500) | 0.888 | 0.836 | +0.052 [0.028, 0.076] |
| USPTO (489) | 0.730 | 0.789 | -0.059 [-0.110, -0.016] |
| JPO (449) | 0.800 | 0.530 | +0.269 [0.220, 0.318] |
| BioVista (1000) | 0.760 | 0.403 | +0.357 [0.323, 0.392] |

CLEF and USPTO exclude 46 and 11 images whose ground truth RDKit cannot parse. CLEF, UOB and USPTO are seeded random subsets of 500 images; JPO is complete; BioVista is a 1,000-image subset.
Notes: MolNexTR's USPTO advantage comes from stereochemistry (0.618 vs 0.412 on structures with stereo); about 47% of the BioVista sample is Markush (0.739 vs 0.126). The "plain /
chiral / markush" split is this project's own heuristic, not BioVista's labelling.

## Running on another machine (no Slurm)

Two Python environments are needed because MolNexTR pins Python 3.8-era packages. **This route has not been tested outside Explorer.**

```bash
# main environment (Python 3.10): the framework and MolGlyph
python3.10 -m venv .venv && . .venv/bin/activate
pip install torch==2.6.0 torchvision==0.21.0            # choose the wheel index that matches your CUDA, or the CPU build
pip install -r requirements.txt -e .
export HF_TOKEN=...                                     # Hugging Face READ token; accept the MolGlyph terms first

# MolNexTR environment (Python 3.8)
git clone --depth 1 https://github.com/CYF2000127/MolNexTR.git ocsr_work/data/_third_party/MolNexTR
python3.8 -m venv .venv-molnextr
.venv-molnextr/bin/pip install torch==2.4.1 -r ocsr_work/data/_third_party/MolNexTR/requirements.txt pystow
export OCSR_MOLNEXTR_PYTHON="$PWD/.venv-molnextr/bin/python"

python -m ocsr_bench selfcheck
python -m ocsr_bench prefetch --config configs/default.yaml
python -m ocsr_bench run --config configs/default.yaml --set run.max_samples=20 --set run.name=smoke_test   # quick check
python -m ocsr_bench run --config configs/default.yaml
```

Data goes to `ocsr_work/` and results to `results/` unless `OCSR_WORK_DIR` / `OCSR_RESULTS_DIR` are set. To skip MolNexTR (no Python 3.8 available) add `--only-tools biominer_molglyph`.

## Status and known limitations

**Verified on Explorer:** compute nodes can reach Hugging Face, Google Drive and GitHub; both environments install; both models load on CPU (prefetch) and run on a GPU; all five datasets load with the layouts
described above; the self-resubmitting job script and the `courses` partitions were **not** exercised (the run used `short` and `gpu`).

1. **The first setup run stopped at its final import check** because MolNexTR's code needs `pystow`, which its `requirements.txt` omits. The package was installed by hand; `setup_envs.sbatch` now
   installs it and repairs further missing modules automatically, but **that updated script has not been re-run from scratch**.
2. **Torch build.** The reported results used torch 2.6.0 + CUDA 12.4 (default in `slurm/env.sh`) and torch 2.4.1 + CUDA 12.1 for MolNexTR. A V100 node (driver 545, CUDA 12.3) was seen but never
   used for the benchmark; on such nodes the CUDA 11.8 build may be needed (see `env.sh`; untested). The exact package versions of the run are in `requirements.lock.txt` and `run_metadata.json`.
3. **Confidence.** MolGlyph returns none (its interface returns SMILES only), so those columns are empty. For MolNexTR the saved value is the mean of per-atom confidences: the worker looks for the
   key `bonds_sets` but MolNexTR returns `bond_sets`, so bond values are ignored. The ordinary accuracy results are unaffected; the confidence analysis would need a rerun after fixing the key.
4. **Tool outputs.** The raw files keep the SMILES, MolGlyph's raw model string and post-processing errors, MolNexTR's confidence and the names of the fields it returned; MolNexTR's molfile and
   atom/bond lists are not stored.
5. **Scope.** BioMiner's full literature-mining system (PDF parsing plus a large language model) needs far more GPU memory than was available; its structure-recognition model, MolGlyph, is benchmarked.
   MolParser (weights not confirmed public), WildMol-10K (no verified download) and the course PDFs (later phases) are not included; OSRA is implemented but untested and disabled.
6. **Metrics.** Exact match compares canonical isomeric SMILES and may differ from BioVista's own scoring script and from MolNexTR's published metric; no claim is made about reproducing MolNexTR's paper.
7. **Course accounts.** The `courses` / `courses-gpu` limits (one running job, 12 GB RAM, 8 h) come from the Course Guide and were not exercised. The regular `short` and `gpu` partitions
   showed 2-day and 8-hour limits in `sinfo`.

## Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `sbatch: error: ... User's group not permitted to use this partition` | You used a bare `sbatch` on a script (course partition default) or your account lacks that partition. Use `bash slurm/submit.sh <job>` with `OCSR_*_PARTITION` set for your account. |
| `sbatch: error: invalid partition` / account or QOS error | Your account may not have that partition: `courses`/`courses-gpu` need classroom access; research accounts use `short`/`gpu` (set the `OCSR_*` variables in `slurm/settings.sh`). Ask rchelp@northeastern.edu. |
| Job killed with an out-of-memory message | `courses-gpu` allows 12 GB: see limitation 2 above. |
| Setup or prefetch stops with `No module named 'x'` for MolNexTR | Install it into its environment and rerun: `$OCSR_ENV_MOLNEXTR/bin/python -m pip install x` (the setup job does this automatically for the import check). |
| Job fails at once and `logs/` is empty | `logs/` did not exist when submitting; use `slurm/submit_all.sh` (it creates it) or `mkdir -p logs`. |
| `module: unable to locate modulefile` | Names differ: edit `OCSR_MODULES` in `slurm/env.sh` after `module avail`. |
| `token=MISSING` in the job log | `~/.hf_token` is missing or unreadable by the job; or export `HF_TOKEN`. |
| 403 on the MolGlyph checkpoint | Accept the model's terms with the same Hugging Face account that issued the token. |
| `python_executable does not exist` | The MolNexTR environment is not built: run `slurm/setup_envs.sbatch` (or `FORCE=1` to rebuild). |
| Disk quota exceeded in `$HOME` | A cache escaped `slurm/env.sh`; run `du -shc ~/.[^.]* ~/*` on a compute node and move it to scratch. |
| A job logs `Run '...'` and then nothing for many minutes (CPU time stays near zero in `sstat -j <id>.batch --format=AveCPU,MaxDiskRead`) | It is waiting on slow filesystem metadata on that node. Rerun on another node: `bash slurm/submit.sh benchmark --exclude=<node>`, and run `bash slurm/submit.sh prefetch` once so the sample cache exists. |
| Job ends at the time limit | Expected: it resubmits itself (up to `MAX_RESUBMITS`, default 10) and resumes. |
| Results disappeared | They were under `/scratch` (purged monthly). Keep `OCSR_RESULTS_DIR` in `$HOME` or `/projects`. |

## References

* BioMiner (MIT): <https://github.com/jiaxianyan/BioMiner>; MolGlyph checkpoint: <https://huggingface.co/jiaxianustc/MolGlyph> (gated).
* MolNexTR: Chen et al., J. Cheminform. 16, 141 (2024); code <https://github.com/CYF2000127/MolNexTR>; checkpoint and test sets: <https://huggingface.co/datasets/CYF200127/MolNexTR>.
* Explorer documentation used for the job design: [Course Guide](https://rc-docs.northeastern.edu/en/latest/classroom/class_guide.html),
  [Connecting to Explorer](https://rc-docs.northeastern.edu/en/latest/connectingtocluster/index.html), [Running Jobs](https://rc-docs.northeastern.edu/en/latest/runningjobs/index.html),
  [Interactive and Batch Mode](https://rc-docs.northeastern.edu/en/latest/runningjobs/interactiveandbatch.html),
  [GPU Access](https://rc-docs.northeastern.edu/en/latest/gpus/accessinggpus.html),
  [GPU Job Submission](https://rc-docs.northeastern.edu/en/latest/gpus/gpujobsubmission.html),
  [Home quota and Conda environments](https://rc-docs.northeastern.edu/en/latest/best-practices/homequota.html),
  [Scratch purge](https://rc-docs.northeastern.edu/en/latest/best-practices/scratchpurge.html),
  [Cluster usage](https://rc-docs.northeastern.edu/en/latest/best-practices/clusterusage.html).
