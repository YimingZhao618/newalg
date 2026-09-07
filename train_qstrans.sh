# #!/bin/bash
# # terran10
# CUDA_VISIBLE_DEVICES="1" python -u src/main.py \
# --config=hpn_qstrans \
# --env-config=sc2_v2_terran_10_vs_10 \
# with \
# obs_agent_id=True \
# obs_last_action=False \
# runner=parallel \
# batch_size_run=8 \
# buffer_size=5000 \
# t_max=10050000 \
# batch_size=128 \
# td_lambda=0.6 \
# crash_model=True \
# t_crash=2000000 \
# rec_explicit_ratio=0.8 \
# epsilon_anneal_time=500000 \
# epsilon_finish=0.06 \
# lr=0.002 \
# crash_mode='normal' \
# name='qstrans_early_crash_terran_10v10'

# # zerg10
# CUDA_VISIBLE_DEVICES="1" python -u src/main.py \
# --config=hpn_qstrans \
# --env-config=sc2_v2_zerg_10_vs_10 \
# with \
# obs_agent_id=True \
# obs_last_action=False \
# runner=parallel \
# batch_size_run=8 \
# buffer_size=5000 \
# t_max=10050000 \
# batch_size=128 \
# td_lambda=0.6 \
# crash_model=True \
# t_crash=2000000 \
# rec_explicit_ratio=0.8 \
# epsilon_anneal_time=500000 \
# epsilon_finish=0.06 \
# lr=0.002 \
# crash_mode='normal' \
# name='qstrans_early_crash_zerg_10v10'

# protoss10
# CUDA_VISIBLE_DEVICES="1" python -u src/main.py \
# --config=hpn_qstrans \
# --env-config=sc2_v2_protoss_10_vs_10 \
# with \
# obs_agent_id=True \
# obs_last_action=False \
# runner=parallel \
# batch_size_run=8 \
# buffer_size=5000 \
# t_max=10050000 \
# batch_size=128 \
# td_lambda=0.6 \
# crash_model=True \
# t_crash=2000000 \
# rec_explicit_ratio=0.8 \
# epsilon_anneal_time=500000 \
# epsilon_finish=0.06 \
# lr=0.002 \
# crash_mode='normal' \
# name='qstrans_early_crash_protoss_10v10'

# terran15
# CUDA_VISIBLE_DEVICES="1" python -u src/main.py \
# --config=hpn_qstrans \
# --env-config=sc2_v2_terran_15_vs_15 \
# with \
# obs_agent_id=True \
# obs_last_action=False \
# runner=parallel \
# batch_size_run=8 \
# buffer_size=5000 \
# t_max=10050000 \
# batch_size=128 \
# td_lambda=0.6 \
# crash_model=True \
# t_crash=2000000 \
# rec_explicit_ratio=0.8 \
# epsilon_anneal_time=500000 \
# epsilon_finish=0.06 \
# lr=0.002 \
# crash_mode='normal' \
# name='qstrans_early_crash_terran_15v15'

# zerg15
# CUDA_VISIBLE_DEVICES="1" python -u src/main.py \
# --config=hpn_qstrans \
# --env-config=sc2_v2_zerg_15_vs_15 \
# with \
# obs_agent_id=True \
# obs_last_action=False \
# runner=parallel \
# batch_size_run=8 \
# buffer_size=5000 \
# t_max=10050000 \
# batch_size=128 \
# td_lambda=0.6 \
# crash_model=True \
# t_crash=2000000 \
# rec_explicit_ratio=0.8 \
# epsilon_anneal_time=500000 \
# epsilon_finish=0.06 \
# lr=0.002 \
# crash_mode='normal' \
# name='qstrans_early_crash_zerg_15v15'

# # protoss15
# CUDA_VISIBLE_DEVICES="1" python -u src/main.py \
# --config=hpn_qstrans \
# --env-config=sc2_v2_protoss_15_vs_15 \
# with \
# obs_agent_id=True \
# obs_last_action=False \
# runner=parallel \
# batch_size_run=8 \
# buffer_size=5000 \
# t_max=10050000 \
# batch_size=128 \
# td_lambda=0.6 \
# crash_model=True \
# t_crash=2000000 \
# rec_explicit_ratio=0.8 \
# epsilon_anneal_time=500000 \
# epsilon_finish=0.06 \
# lr=0.002 \
# crash_mode='normal' \
# name='qstrans_early_crash_protoss_15v15'

# # terran20
CUDA_VISIBLE_DEVICES="2" python -u src/main.py \
--config=hpn_qstrans \
--env-config=sc2_v2_terran_20_vs_20 \
with \
obs_agent_id=True \
obs_last_action=False \
runner=parallel \
batch_size_run=8 \
buffer_size=5000 \
t_max=10050000 \
batch_size=128 \
td_lambda=0.6 \
crash_model=True \
t_crash=5000000 \
rec_explicit_ratio=0.8 \
epsilon_anneal_time=500000 \
epsilon_finish=0.06 \
lr=0.002 \
crash_mode='crash_temp' \
name='qstrans_temp_crash_terran_20v20'

#zerg20
CUDA_VISIBLE_DEVICES="2" python -u src/main.py \
--config=hpn_qstrans \
--env-config=sc2_v2_zerg_20_vs_20 \
with \
obs_agent_id=True \
obs_last_action=False \
runner=parallel \
batch_size_run=8 \
buffer_size=5000 \
t_max=10050000 \
batch_size=128 \
td_lambda=0.6 \
crash_model=True \
t_crash=5000000 \
rec_explicit_ratio=0.8 \
epsilon_anneal_time=500000 \
epsilon_finish=0.06 \
lr=0.002 \
crash_mode='crash_temp' \
name='qstrans_temp_crash_zerg_20v20'

#protoss20
CUDA_VISIBLE_DEVICES="1" python -u src/main.py \
--config=hpn_qstrans \
--env-config=sc2_v2_protoss_20_vs_20 \
with \
obs_agent_id=True \
obs_last_action=False \
runner=parallel \
batch_size_run=8 \
buffer_size=5000 \
t_max=10050000 \
batch_size=128 \
td_lambda=0.6 \
crash_model=True \
t_crash=5000000 \
rec_explicit_ratio=0.8 \
epsilon_anneal_time=500000 \
epsilon_finish=0.06 \
lr=0.002 \
crash_mode='crash_temp' \
name='qstrans_temp_crash_protoss_20v20'