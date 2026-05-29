import re
import sys
from datetime import datetime, timezone

faith = sys.argv[1] if len(sys.argv) > 1 else "n/a"
relev = sys.argv[2] if len(sys.argv) > 2 else "n/a"
prec  = sys.argv[3] if len(sys.argv) > 3 else "n/a"
now   = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

scores = (
    f"**RAGAS Evaluation** — updated {now}\n\n"
    f"| Metric | Score |\n"
    f"|---|---|\n"
    f"| Faithfulness | {faith} |\n"
    f"| Answer Relevancy | {relev} |\n"
    f"| Context Precision | {prec} |\n"
    f"\n_Higher is better (0.0 – 1.0). Run `python evaluate.py` to re-score on your own document._\n"
)

with open("README.md") as f:
    content = f.read()

content = re.sub(r"<!-- EVAL_SCORES -->.*?<!-- END_EVAL_SCORES -->", f"<!-- EVAL_SCORES -->\n{scores}<!-- END_EVAL_SCORES -->", content, flags=re.DOTALL)

with open("README.md", "w") as f:
    f.write(content)
