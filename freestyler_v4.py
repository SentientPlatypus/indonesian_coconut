"""
V4 trainer — scoring-first, with air-dribble & flip-reset curriculum.

This is the training "hand" the self-improving /loop drives. It is also runnable
standalone (uses default config + resume-from-latest).

The reward set, weights, curriculum mix, and ent_coef all come from a CONFIG dict
(see loop_config.py); the env itself is built in v4_env.py so trainer and
evaluator can never drift. The /loop edits the config JSON between phases.

Controlled by env vars (all optional):
  V4_LOOP_CONFIG : path to the config JSON   (default: built-in DEFAULT_CONFIG)
  V4_RESUME_DIR  : checkpoint dir to resume from. The loop points this at the
                   current BEST checkpoint, so a rejected candidate is discarded
                   simply by resuming from best again.
                   (default: latest V4 sub-checkpoint, else V3 17.9B)
  V4_SAVE_DIR    : where to save           (default: data/checkpoints/V4)
  V4_N_PROC      : RocketSim collector procs (default 28, sized for g6.8xlarge
                   = 32 vCPU / 1x L4; scales with CPUs, not GPUs — raise toward
                   30 if steps/sec is still climbing, drop if it plateaus)
  V4_WANDB       : "0" to DISABLE Weights & Biases (default ON). Requires a
                   one-time `wandb login` on the box, else the run errors out.
  V4_WANDB_PROJECT : wandb project      (default "indococo-v4-loop")
  V4_WANDB_GROUP   : wandb group; the loop sets one stable name per session so
                     all iterations stack in one view (default: unset)
  V4_WANDB_RUN     : wandb run name; the loop sets "iter_<N>" per phase so each
                     eval-gated phase is its own line (default: wandb auto-name)

The loop runs this in the background and stops it after a phase, then evaluates
the latest saved checkpoint. Checkpoints carry optimizer state, so stop/resume
is lossless.

Reward design, the GoalDistReward fix, and the curriculum rationale are
documented in loop_config.py / curriculum_mutators.py / v4_env.py.
"""
import os

V3_FALLBACK = "data/checkpoints/V3/17.9B"   # first-ever start point (the 47-3 policy)
DEFAULT_SAVE_DIR = "data/checkpoints/V4"


def _latest_subcheckpoint(folder):
    """Newest numbered sub-checkpoint inside a save folder, or None if empty/missing."""
    if not folder or not os.path.isdir(folder):
        return None
    subs = [d for d in os.listdir(folder) if d.isdigit()]
    if not subs:
        return None
    return os.path.join(folder, max(subs, key=int))


def _resolve_resume_dir(save_dir):
    """V4_RESUME_DIR if set, else latest V4 checkpoint, else the V3 fallback."""
    explicit = os.environ.get("V4_RESUME_DIR")
    if explicit:
        return explicit
    latest_v4 = _latest_subcheckpoint(save_dir)
    if latest_v4:
        return latest_v4
    return _latest_subcheckpoint(V3_FALLBACK)


def build_rlgym_v2_env():
    # Zero-arg factory for rlgym_ppo (called in each subprocess; reads the same
    # config file via the inherited V4_LOOP_CONFIG env var).
    from loop_config import load_config
    from v4_env import build_env
    cfg = load_config(os.environ.get("V4_LOOP_CONFIG"))
    return build_env(cfg, for_training=True)


if __name__ == "__main__":
    from rlgym_ppo import Learner
    from loop_config import load_config

    cfg = load_config(os.environ.get("V4_LOOP_CONFIG"))
    save_dir = os.environ.get("V4_SAVE_DIR", DEFAULT_SAVE_DIR)
    resume_dir = _resolve_resume_dir(save_dir)
    print(f"[V4] resume_dir={resume_dir}")
    print(f"[V4] save_dir={save_dir}")
    print(f"[V4] ent_coef={cfg['ppo_ent_coef']} curriculum={cfg['curriculum']}")

    n_proc = int(os.environ.get("V4_N_PROC", "28"))   # 32-vCPU g6.8xlarge; leaves ~4 for the learner + GPU feed
    min_inference_size = max(1, int(round(n_proc * 0.9)))

    log_wandb = os.environ.get("V4_WANDB", "1") != "0"   # ON by default now

    learner = Learner(
        build_rlgym_v2_env,
        n_proc=n_proc,
        checkpoint_load_folder=resume_dir,
        checkpoints_save_folder=save_dir,
        add_unix_timestamp=False,            # stable, discoverable path the loop reads
        min_inference_size=min_inference_size,
        metrics_logger=None,
        ppo_batch_size=100_000,
        # Layer sizes MUST match the 17.9B checkpoint to resume — do not change.
        policy_layer_sizes=[2048, 2048, 1024, 1024],
        critic_layer_sizes=[2048, 2048, 1024, 1024],
        ts_per_iteration=100_000,
        exp_buffer_size=300_000,
        ppo_minibatch_size=50_000,
        ppo_ent_coef=cfg["ppo_ent_coef"],    # tunable knob (loop may nudge it)
        policy_lr=1e-4,
        critic_lr=1e-4,
        ppo_epochs=2,
        standardize_returns=True,
        standardize_obs=False,
        save_every_ts=1_000_000,
        timestep_limit=10_000_000_000_000,   # effectively unbounded; the loop stops it
        log_to_wandb=log_wandb,
        # load_wandb=False: each phase resumes from the BEST checkpoint but is a
        # FRESH wandb run, so we don't reload/append the run id baked into an old
        # checkpoint. Group + run name keep iterations comparable in one view.
        load_wandb=False,
        wandb_project_name=os.environ.get("V4_WANDB_PROJECT", "indococo-v4-loop"),
        wandb_group_name=os.environ.get("V4_WANDB_GROUP"),
        wandb_run_name=os.environ.get("V4_WANDB_RUN"),
        render=False,                        # EC2 training — no display
    )
    learner.learn()
