"""
Mono2Micro Pipeline — Spring PetClinic
Orchestrates extraction → feature engineering → similarity matrix → clustering → evaluation
"""
import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")

STEPS = ["extract", "features", "similarity", "cluster", "evaluate", "all"]


def run_extract(cfg):
    from src.extractor.java_extractor import JavaExtractor
    ex = JavaExtractor(cfg["src_root"], cfg["output_dir"])
    ex.run()


def run_features(cfg):
    from src.features.feature_builder import FeatureBuilder
    fb = FeatureBuilder(cfg["output_dir"])
    fb.run()


def run_similarity(cfg):
    from src.features.similarity_matrix import SimilarityMatrixBuilder
    sb = SimilarityMatrixBuilder(cfg["output_dir"], weights=cfg.get("weights"))
    sb.run()


def run_cluster(cfg):
    from src.clustering.clusterer import Clusterer
    c = Clusterer(cfg["output_dir"], n_clusters=cfg.get("n_clusters", 5))
    c.run()


def run_evaluate(cfg):
    from src.evaluation.evaluator import Evaluator
    e = Evaluator(cfg["output_dir"], cfg.get("ground_truth_csv"))
    e.run()


def main():
    parser = argparse.ArgumentParser(description="Mono2Micro Pipeline for Spring PetClinic")
    parser.add_argument("step", choices=STEPS, help="Pipeline step to execute")
    parser.add_argument("--src", default="spring-petclinic/src/main/java",
                        help="Path to PetClinic Java sources")
    parser.add_argument("--output", default="outputs", help="Output directory")
    parser.add_argument("--ground-truth", default=None, help="Path to ground_truth.csv")
    parser.add_argument("--n-clusters", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.35, help="Structural weight")
    parser.add_argument("--beta",  type=float, default=0.30, help="Data-access weight")
    parser.add_argument("--gamma", type=float, default=0.20, help="Semantic weight")
    parser.add_argument("--delta", type=float, default=0.15, help="Behavioral weight")
    args = parser.parse_args()

    cfg = {
        "src_root":       args.src,
        "output_dir":     args.output,
        "ground_truth_csv": args.ground_truth,
        "n_clusters":     args.n_clusters,
        "weights": {
            "alpha": args.alpha,
            "beta":  args.beta,
            "gamma": args.gamma,
            "delta": args.delta,
        },
    }

    dispatch = {
        "extract":    run_extract,
        "features":   run_features,
        "similarity": run_similarity,
        "cluster":    run_cluster,
        "evaluate":   run_evaluate,
    }

    if args.step == "all":
        for step_name, fn in dispatch.items():
            log.info("═══ STEP: %s ═══", step_name.upper())
            fn(cfg)
    else:
        dispatch[args.step](cfg)

    log.info("Done.")


if __name__ == "__main__":
    main()
