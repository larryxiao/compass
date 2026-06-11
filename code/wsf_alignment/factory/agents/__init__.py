"""Multi-agent factory for self-improving dilemma generation.

Four roles:
  - setup    — generates new dilemma candidates (prompt is the meta-learning surface)
  - decide   — answers a dilemma cold, no options shown (free-text reasoning)
  - evaluate — judges intrinsic quality on a 5-dimension rubric
  - refine   — picks top-K, rewrites the setup-agent prompt for next iteration
"""
