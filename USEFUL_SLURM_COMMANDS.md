### 1. Quickest: Show all idle nodes in your partition
```bash
sinfo -p u22 -t idle
```
*(This gives a list of nodes that are 100% free with no jobs currently running).*

---

### 2. Best for GPUs: Show idle nodes with their GPU models
To see which specific GPU nodes (like `2080ti` / `3080`) are idle right now:
```bash
sinfo -N -p u22 -t idle -O NodeList,Gres,Features,StateComplete
```

---

### 3. Check Partially Free Nodes (`mix`)
Often, a node has 4 GPUs, but someone is only using 1 (leaving 3 GPUs free). These nodes show up as `mix` rather than `idle`:
```bash
sinfo -p u22 -t idle,mix
```

---

### 4. Check if a specific GPU type has free nodes right now
To quickly see if any `2080ti` / `3080` nodes are completely idle:
```bash
sinfo -p u22 --constraint=2080ti -t idle
```

### 5. Just get a gnode
To get a shell on a gnode:
```bash
srun -p u22 -A research --qos=medium --gres=gpu:1 --constraint=2080ti --cpus-per-task=8 --mem=32G --pty bash
```
