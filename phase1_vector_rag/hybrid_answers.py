"""Generate Phase 1b (hybrid) answers for the active dataset."""
from shared.hotpot_runner import run_phase
from phase1_vector_rag.hybrid_rag import run_hybrid_rag_pipeline


if __name__ == "__main__":
    run_phase(run_hybrid_rag_pipeline, phase=1, phase_name="Phase 1b (hybrid)")
