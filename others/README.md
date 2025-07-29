# VIVTransformer

This repository contains scripts for training and testing variants of the
VIVTransformer model with multiple attention mechanisms.

## Quick Start

1. Install the required Python packages:

```bash
pip install -r modify_multi_attention/requirements.txt
```

2. Run training with the default configuration:

```bash
python modify_multi_attention/main.py
```

Use `--config` to specify a custom YAML configuration and `--results-dir` to
change where outputs are written.

3. To duplicate the repository without its Git history for experimentation:

```bash
python duplicate_repo.py /path/to/destination
```

