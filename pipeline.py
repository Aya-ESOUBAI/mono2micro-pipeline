"""
Mono2Micro Pipeline — Spring PetClinic
Orchestrates: extraction -> features -> similarity -> clustering -> evaluation
OpenBLAS-free: all steps use pure Python or CSV-only pandas.
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
    JavaExtractor(cfg["src_root"], cfg["output_dir"]).run()


def run_features(cfg):
    from src.features.feature_builder import FeatureBuilder
    FeatureBuilder(cfg["output_dir"]).run()


def run_similarity(cfg):
    from src.features.similarity_matrix import SimilarityMatrixBuilder
    SimilarityMatrixBuilder(cfg["output_dir"], weights=cfg.get("weights")).run()


def run_cluster(cfg):
    from src.clustering.clusterer import Clusterer
    Clusterer(cfg["output_dir"], n_clusters=cfg.get("n_clusters", 5)).run()


def run_evaluate(cfg):
    from src.evaluation.evaluator import Evaluator
    Evaluator(cfg["output_dir"], cfg.get("ground_truth_csv")).run()


def main():
    parser = argparse.ArgumentParser(description="Mono2Micro Pipeline")
    parser.add_argument("step", choices=STEPS)
    parser.add_argument("--src",     default="spring-petclinic/src/main/java")
    parser.add_argument("--output",  default="outputs")
    parser.add_argument("--ground-truth", default=None)
    parser.add_argument("--n-clusters",   type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--beta",  type=float, default=0.30)
    parser.add_argument("--gamma", type=float, default=0.20)
    parser.add_argument("--delta", type=float, default=0.15)
    args = parser.parse_args()

    cfg = {
        "src_root":         args.src,
        "output_dir":       args.output,
        "ground_truth_csv": args.ground_truth,
        "n_clusters":       args.n_clusters,
        "weights": {
            "alpha": args.alpha, "beta": args.beta,
            "gamma": args.gamma, "delta": args.delta,
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
        for name, fn in dispatch.items():
            log.info("=== STEP: %s ===", name.upper())
            fn(cfg)
    else:
        dispatch[args.step](cfg)

    log.info("Done.")


if __name__ == "__main__":
    main()
