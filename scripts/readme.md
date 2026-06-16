conda activate isaac_rm
cd /home/hjc/isaac_rm
OMNI_KIT_ACCEPT_EULA=YES python scripts/play.py \
  --task Isaac-Chassis-Approach-Play-v0 --num_envs 4 \
  --checkpoint /home/hjc/isaac_rm/logs/skrl/chassis_approach/2026-06-15_21-24-36_ppo_torch/checkpoints/best_agent.pt

OMNI_KIT_ACCEPT_EULA=YES python scripts/play.py \
  --task Isaac-Chassis-Approach-Play-v0 --num_envs 4 --headless --video --video_length 300 \
  --checkpoint /home/hjc/isaac_rm/logs/skrl/chassis_approach/2026-06-15_23-27-11_ppo_torch/checkpoints/best_agent.pt
