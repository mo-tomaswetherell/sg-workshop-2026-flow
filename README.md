# sg-workshop-2026-flow
Flow matching demo for the CCRS workshop.

## Setup (ASPIRE2A)

> On ASPIRE2A, only the environment setup is done on the login node. Running the
> notebooks (training/inference) must happen on a **GPU compute node** requested
> through the PBS scheduler — never on the login node.

### 1. Activate the conda environment (login node)

After connecting to ASPIRE2A, load conda. Then activate the environment required for this demo and register it as a Jupyter kernel so the
`sg-fm-demo` kernel the notebooks expect appears in JupyterLab:

```bash
module load miniforge3
conda activate /home/users/industry/metoffice/04052003/.conda/envs/sg-fm-demo
python -m ipykernel install --user --name sg-fm-demo --display-name sg-fm-demo
```

> Note: do not run `conda init` or add it to `~/.bashrc` on ASPIRE2A — NSCC
> advises against it as it can slow down logins.

### 2. Request a GPU node and start JupyterLab

This repo includes a ready-made batch script, [`jupyter.pbs`](jupyter.pbs). Submit it and check the job:

```bash
qsub jupyter.pbs
qstat @pbs102
```

The script requests one GPU, activates the `sg-fm-demo` environment, and starts
JupyterLab on the compute node.

### 3. Connect from your laptop

When you submit the job with `qsub`, PBS prints a job ID (e.g.
`12345.pbs101`). When the job starts running, the script creates two text files
in the directory you submitted from (on the login node): `sshtunnel.<jobid>`
(the tunnel command) and `jpylab.<jobid>` (JupyterLab's log, including the URL).

1. **On the login node**, print the tunnel command and copy it:

   ```bash
   cat sshtunnel.12345.pbs101    # use your actual job ID
   ```

   This shows one line, for example:

   ```bash
   ssh -N -L 8541:asp2a-gpu003:8541 yourId@aspire2a.nscc.sg
   ```

2. In a **separate terminal that is *not* logged into the HPC** (a plain local
   shell), paste and run that command. It opens the tunnel by logging in for
   you, so don't SSH in beforehand. It will appear to hang with no output —
   that's expected; leave it running.

3. Find JupyterLab's URL in the log and open it in your local browser:

   ```bash
   cat jpylab.12345.pbs101       # on the login node
   ```

   Copy the `http://127.0.0.1:<PORT>/lab?token=...` line into your browser.

Then open [`notebooks/training.ipynb`](notebooks/training.ipynb) or
[`notebooks/inference.ipynb`](notebooks/inference.ipynb) and select the
`sg-fm-demo` kernel.

> For quick testing/debugging you can instead grab an interactive GPU session
> with `qsub -I -l select=1:ngpus=1 -l walltime=1:00:00 -q ai -P <projectId>`
> and launch `jupyter lab` from there.

4. At the end of the demo, save your notebooks. In the tunnel terminal 
   (from step 2), end the connection:
      
      ```bash
   ctrl+c
   ```

   In the login node terminal:
      ```bash
   qdel <jobid>
   ```

---

See the [codebase](https://github.com/mo-tomaswetherell/mo-flow-sg) associated with our paper 'Flow Matching for Convective-Scale Precipitation Downscaling'. 
