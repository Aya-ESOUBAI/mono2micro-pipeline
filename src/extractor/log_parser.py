"""
src/extractor/log_parser.py
────────────────────────────
Parses real Spring Boot / Hibernate logs produced by:

  logging.level.org.springframework.web=DEBUG
  logging.level.org.hibernate.SQL=DEBUG

and produces outputs/logs_features.json with:
  - per-class endpoint_count, http_resource
  - request traces (list of class lists for co-occurrence matrix)

Usage:
  python -c "
  from src.extractor.log_parser import LogParser
  LogParser('app.log', 'outputs').run()
  "

Or call from the pipeline:
  from src.extractor.log_parser import LogParser
  LogParser(log_path, output_dir).run()
"""

import json
import logging
import re
from collections import defaultdict
from pathlib import Path

log = logging.getLogger(__name__)

# ── regex patterns ─────────────────────────────────────────────────────────────
HTTP_REQUEST_PAT = re.compile(
    r'(GET|POST|PUT|DELETE|PATCH)\s+(/[\w/{}?=&]+)',
)
CLASS_INVOKE_PAT = re.compile(
    r'(?:Mapped to|Executing|Processing).*?(\w+(?:Controller|Service|Repository))',
)
SQL_PAT = re.compile(
    r'(?:select|insert|update|delete).*?\bfrom\s+(\w+)',
    re.IGNORECASE,
)


class LogParser:
    def __init__(self, log_path: str, output_dir: str):
        self.log_path   = Path(log_path)
        self.output_dir = Path(output_dir)

    def run(self):
        if not self.log_path.exists():
            log.warning("Log file not found: %s — generating minimal placeholder", self.log_path)
            self._write_placeholder()
            return

        log.info("Parsing log: %s", self.log_path)
        text = self.log_path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()

        traces: list[dict] = []
        current_trace: dict | None = None

        endpoint_counts: dict[str, int] = defaultdict(int)
        class_endpoints: dict[str, set[str]] = defaultdict(set)

        for line in lines:
            # new request → new trace
            http_m = HTTP_REQUEST_PAT.search(line)
            if http_m:
                if current_trace and current_trace["classes"]:
                    traces.append(current_trace)
                method, path = http_m.group(1), http_m.group(2)
                endpoint = f"{method} {path}"
                endpoint_counts[endpoint] += 1
                current_trace = {
                    "endpoint": endpoint,
                    "classes":  [],
                }

            # class invoked within trace
            cls_m = CLASS_INVOKE_PAT.search(line)
            if cls_m and current_trace is not None:
                cls_name = cls_m.group(1)
                current_trace["classes"].append(cls_name)
                class_endpoints[cls_name].add(current_trace["endpoint"])

        if current_trace and current_trace["classes"]:
            traces.append(current_trace)

        # per-class summary
        class_features = [
            {
                "class":          cls,
                "endpoint_count": len(eps),
                "endpoints":      sorted(eps),
            }
            for cls, eps in class_endpoints.items()
        ]

        out = {
            "traces":         traces,
            "class_features": class_features,
            "endpoint_counts": dict(endpoint_counts),
        }
        (self.output_dir / "logs_features.json").write_text(json.dumps(out, indent=2))
        log.info("logs_features.json: %d traces, %d class entries",
                 len(traces), len(class_features))

    def _write_placeholder(self):
        """
        Minimal placeholder so the pipeline doesn't crash.
        Replace with real JMeter / k6 traces.
        """
        placeholder = {
            "traces": [],
            "class_features": [],
            "endpoint_counts": {},
            "_warning": (
                "No real log file found. "
                "Run PetClinic with DEBUG logging + a JMeter scenario, "
                "then call LogParser('app.log', 'outputs').run() again."
            ),
        }
        (self.output_dir / "logs_features.json").write_text(
            json.dumps(placeholder, indent=2)
        )
        log.info("Wrote placeholder logs_features.json")


# ── ground truth helper ───────────────────────────────────────────────────────

def write_ground_truth(output_dir: str, fqns: list[str]):
    """
    Generates ground_truth.csv mapping each class FQN to its target microservice
    based on the spring-petclinic-microservices reference implementation.
    """
    from src.evaluation.evaluator import BUILTIN_GROUND_TRUTH, SERVICE_TO_ID

    rows = []
    for fqn in fqns:
        simple = fqn.split(".")[-1]
        svc = BUILTIN_GROUND_TRUTH.get(simple, "unknown")
        rows.append({"fqn": fqn, "service": svc, "service_id": SERVICE_TO_ID.get(svc, 5)})

    import pandas as pd
    df = pd.DataFrame(rows)
    out = Path(output_dir) / "ground_truth.csv"
    df.to_csv(out, index=False)
    log.info("ground_truth.csv: %d entries", len(rows))
    return str(out)
