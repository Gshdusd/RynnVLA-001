[![Releases](https://img.shields.io/badge/Releases-Download-blue?logo=github&style=for-the-badge)](https://github.com/Gshdusd/RynnVLA-001/releases)

# RynnVLA-001: Vision-Language-Action Model with Generative Priors

🚀 A compact, modular implementation of a vision-language-action (VLA) agent. RynnVLA-001 fuses visual encoders, language models, and action policies with generative priors to improve planning and zero-shot generalization.

![RynnVLA Hero](https://images.unsplash.com/photo-1518770660439-4636190af475?auto=format&fit=crop&w=1400&q=80)

Table of contents
- About
- Key features
- Releases
- Quick start
- Model components
- Training and data
- Inference and API
- Evaluation
- Reproducibility
- Examples
- Contribution guide
- License
- Citation
- Acknowledgments

About
RynnVLA-001 couples a vision encoder, a language backbone, a generative prior module, and an action policy head. The prior shapes latent representations with learned generative constraints. The policy uses latent space and text context to produce action sequences. The design fits embodied agents, robotic control in simulated environments, and multimodal planning tasks.

Key features
- Multimodal fusion: visual features and language context fuse at the latent level.
- Generative priors: variational or diffusion priors regularize latent dynamics.
- Action policy: autoregressive and transformer-based policy heads.
- Modular code: swap encoders, priors, or policy modules.
- Small footprint: prebuilt release artifacts for local runs and demos.
- Reproducible scripts: training, eval, and inference scripts included.

Releases
Download the release artifact and execute it. The release page contains binaries, model checkpoints, and runnable examples. Go to:
https://github.com/Gshdusd/RynnVLA-001/releases

Follow the release page and download the release file for your platform. After download, unpack the archive and run the provided start script or binary. The release asset typically contains:
- model checkpoints (.pt / .pth)
- CLI binary or Python package
- demo scripts and configs

Quick start

Requirements
- Linux or macOS (tested)
- Python 3.9+
- CUDA 11.7 or compatible (for GPU runs)
- 16 GB RAM recommended for training; 8 GB for small demos

Install from source (dev)
1. Clone the repo
   git clone https://github.com/Gshdusd/RynnVLA-001.git
   cd RynnVLA-001
2. Create venv and install
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

Run a demo using the release file
1. Visit the release page and download the main release asset:
   https://github.com/Gshdusd/RynnVLA-001/releases
2. Unpack the file you downloaded and run the start script:
   tar -xzf rynnvla-release.tar.gz
   cd rynnvla-release
   ./run_demo.sh
3. The demo loads a small visual dataset, a lightweight prior, and a policy. The demo prints a short action plan and an image overlay showing predicted intent.

Quick CLI examples
- Run inference on a single image + prompt:
  python scripts/infer.py --image assets/task1.png --prompt "open the red drawer" --checkpoint checkpoints/latest.pt
- Train a small model (debug mode):
  python train.py --config configs/debug.yaml --max-steps 1000
- Evaluate on holdout set:
  python eval.py --split val --checkpoint checkpoints/latest.pt

Model components

Visual encoder
- Backbone: ResNet-50 or a CLIP ViT variant
- Output: dense visual tokens and global embeddings
- Use cases: object-level features, spatial attention maps

Language backbone
- Options: small transformer (GPT-like) or off-the-shelf LLM adapter
- Tokenization: BPE or SentencePiece
- Role: encode instructions, goal descriptions, and constraints

Generative prior
- Types: VAE prior, conditional diffusion prior
- Goal: produce structured latent trajectories that match plausible scene changes
- Benefit: improves zero-shot policy behavior when facing novel goals

Latent controller and policy
- Latent fusion: cross-attention between language tokens and visual tokens
- Policy head: autoregressive transformer or MLP with recurrence
- Output: discrete action tokens or continuous control vectors

Modularity and interfaces
- Each module exposes a clear interface:
  - encode_image(image) -> visual_latents
  - encode_text(text) -> text_latents
  - prior.sample(condition) -> latent_traj
  - policy.act(latents, state) -> actions
- Swap or replace modules by editing configs. The system loads modules by name.

Architecture diagram
![Architecture diagram](https://upload.wikimedia.org/wikipedia/commons/4/4e/Diagram_of_Transformer_Model.svg)

Training and data

Datasets
- Multi-task simulated data: scripted interactions in MuJoCo or PyBullet
- Vision-language pairs: synthetic captions aligned to images
- Real-world datasets: selective integration of human teleoperation logs
- Data format: TFRecord or JSONL with fields {image, prompt, actions, state}

Losses
- Reconstruction: L2 or perceptual loss for vision reconstructions
- Prior regularization: KL divergence or diffusion loss
- Policy loss: behavior cloning (cross-entropy or MSE) and optional RL fine-tuning
- Auxiliary: contrastive loss for image-text alignment

Training recipe (example)
1. Pretrain the visual encoder (if needed) on image classification or CLIP-style alignment.
2. Pretrain generative prior on latent sequences derived from action trajectories.
3. Train the policy with behavior cloning with frozen priors for stability.
4. Optionally fine-tune full stack end-to-end with a small learning rate.

Hyperparameters (starter)
- batch_size: 64 (or lower for GPU memory)
- learning_rate: 1e-4 for encoder/policy, 5e-5 for language backbone
- prior_lr: 1e-4
- scheduler: cosine with warmup
- epochs: 50 for medium datasets; reduce for prototyping

Checkpoints
- Save periodic checkpoints with optimizer state and config
- Export final model with a small wrapper for inference

Inference and API

Core API (Python)
- from rynnvla import Runner
- runner = Runner.from_checkpoint("checkpoints/latest.pt")
- actions = runner.plan(image="assets/task1.png", prompt="place cup on table", steps=8)
- runner.execute(actions)

REST demo server
- A small Flask or FastAPI app serves inference:
  POST /infer with payload {image, prompt, steps}
  returns actions and optional visualization overlays

Action output formats
- Discrete tokens: ['move_forward', 'grip_close', ...]
- Continuous vectors: joint torques or end-effector deltas
- Visualization: action timelines, predicted frames from prior

Evaluation

Metrics
- Task success rate: final state matches goal predicate
- Edit distance: compare predicted action tokens to ground-truth
- Latent plausibility: evaluate prior sample likelihood on held-out trajectories
- Robustness: test on shifted visuals and unseen prompts

Benchmarks
- Include scripts to run standardized benchmarks in /benchmarks
- Example: run_bench.sh executes tasks with varying visual occlusion and reports metrics

Reproducibility

Config system
- configs/ holds YAML configs for models and experiments
- Each run logs command, git commit hash, and environment

Random seeds
- Set seeds in one place: rynnvla.utils.set_global_seed(seed)
- Report seed in logs

Containers
- A Dockerfile builds an environment for consistent runs
- Example:
  docker build -t rynnvla:latest .
  docker run --gpus all -v $(pwd)/data:/data rynnvla:latest ./run_demo.sh

Examples

Example 1 — Navigation with object goal
- Input: image of a table, prompt "bring the red mug to the left tray"
- Output: sequence of motions and grasp commands. The prior predicts intermediate scene changes.

Example 2 — Tool use
- Input: image with a lever, prompt "pull lever until indicator shows green"
- Output: high-level plan and low-level control adjustments to reach the goal

Scripts
- scripts/demo_nav.sh
- scripts/demo_tool.sh
- Each script loads a small checkpoint and runs a demo trajectory. Check the release for runnable scripts.

Contribution guide
- Fork the repo
- Create a feature branch
- Run unit tests and linters
- Open a pull request with a clear description and tests
- Use the issue tracker to propose major changes or new datasets

Coding style
- Follow PEP8 and type hints for public functions
- Keep functions small and testable
- Add or update docs in docs/ when you change APIs

License
- The project uses an open-source license. See LICENSE for details.

Citation
If you use RynnVLA-001 in research, cite the repository and the model. Example BibTeX:
@misc{rynnvla2025,
  title = {RynnVLA-001: Vision-Language-Action Model with Generative Priors},
  author = {RynnVLA Contributors},
  year = {2025},
  howpublished = {GitHub repository},
  note = {https://github.com/Gshdusd/RynnVLA-001}
}

Acknowledgments
- Thanks to open-source libraries that enable rapid experimentation: PyTorch, Hugging Face, OpenAI CLIP, and ecosystem tools.
- Visual assets via Unsplash and Wikimedia Commons.

Release link (again)
Download the release artifact and execute it:
https://github.com/Gshdusd/RynnVLA-001/releases

Contact
- Open issues on GitHub for bugs or feature requests
- Pull requests welcome for model additions, dataset wrappers, and new priors