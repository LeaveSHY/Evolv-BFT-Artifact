"""List all numerical claims in appendix.tex with surrounding context."""
import re
from pathlib import Path

path = Path("d:/Alex/Papers/NDSS 2027_SUBMISSION/appendix.tex")
lines = path.read_text(encoding="utf-8").splitlines()

# Patterns that indicate a numerical empirical claim
patterns = [
    re.compile(r"\$?[\d.]+\s*\\?%"),                    # 82.4%, $82.4\%$
    re.compile(r"\$?[\d.]+\s*pp"),                       # 8.5pp
    re.compile(r"\$?[\d.]+\s*\\\,?(?:ms|s|x|times|fold)"),
    re.compile(r"\$?[\d.]+\s*\{?[\\\\]times"),
    re.compile(r"=\s*\d+\.\d+"),                        # =99.7
    re.compile(r"=\s*0\.\d+\b"),                        # =0.99, =1.000
    re.compile(r"\$\\\\sim\s*\$?[\d.]+"),
    re.compile(r"\$\\sim"),
    re.compile(r"AUROC"),                               # AUROC mentions
    re.compile(r"TPR\s*[\$=]"),
    re.compile(r"FPR\s*[\$=]"),
    re.compile(r"\d+:\d"),                               # 4:1
    re.compile(r"\d+,\d{3}\s*episodes"),
    re.compile(r"\d+\s*episodes"),
    re.compile(r"\bseeds\b"),
]

interesting = set()
for i, line in enumerate(lines):
    # Skip table data lines, comments
    s = line.strip()
    if s.startswith('%') or not s:
        continue
    for p in patterns:
        if p.search(line):
            interesting.add(i)
            break

# Group consecutive lines into blocks
out = []
prev = -10
for i in sorted(interesting):
    if i - prev > 3:
        out.append([])
    out[-1].append(i)
    prev = i

# Print one block per claim, with line number
for block in out:
    start = min(block)
    end = max(block)
    # Skip pure table lines and figure captions for cleanliness
    snippet = " ".join(lines[i].strip() for i in range(start, end+1))
    # Strip very long
    if len(snippet) > 300:
        snippet = snippet[:280] + "..."
    print(f"L{start+1}-{end+1}: {snippet}")
