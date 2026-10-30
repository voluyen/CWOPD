"""Consequence-weighted on-policy distillation (CW-OPD, proposal.pdf).

On-policy distillation in which each step of the student's rollout is weighted by how much it moves
the teacher's probability of reaching its own final answer, read from answer probes packed into the
teacher forward pass. Runs as its own family (`cwopd run ...`, configs under training_configs/cwopd/).
"""
