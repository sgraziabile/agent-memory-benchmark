# agents.level_2_episodic_rag — Level 2: Episodic Naive RAG agent.
#
# Implements an append-only cross-thread episodic memory using Naive RAG
# (retrieve_node → model_node → append_node) over a swappable
# EpisodicMemoryBackend. The agent graph itself lands in Milestone 4 of
# TODOs/phase-2-l2-vector-implementation-plan.md; Milestone 1 provides the
# backend data contracts under backends/.
