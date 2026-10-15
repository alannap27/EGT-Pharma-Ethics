"""Turn scripts/notebook_src.py into egt_pharma_analysis.ipynb and run it, saving the
printed output and the figures into the notebook. Doesn't need Jupyter installed.
Afterwards the figures are copied into paper/figures so the paper always uses the
current ones.

Run from the repo root:
    python scripts/build_notebook.py scripts/notebook_src.py egt_pharma_analysis.ipynb
"""
import json, sys, io, os, time, base64, contextlib, traceback, re
sys.path.insert(0, os.getcwd())   # run from the repository root so egt_core is importable
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def parse(path):
    cells, cur, kind = [], [], None
    for line in open(path).read().splitlines():
        if line.startswith("# %%"):
            if kind is not None: cells.append((kind, cur))
            kind = "markdown" if "[markdown]" in line else "code"; cur = []
        else:
            cur.append(line)
    if kind is not None: cells.append((kind, cur))
    out = []
    for kind, lines in cells:
        while lines and not lines[-1].strip(): lines.pop()
        while lines and not lines[0].strip(): lines.pop(0)
        if kind == "markdown":
            lines = [l[2:] if l.startswith("# ") else (l[1:] if l.startswith("#") else l) for l in lines]
        out.append((kind, "\n".join(lines)))
    return out

def to_source(text):
    ls = text.split("\n")
    return [l + "\n" for l in ls[:-1]] + [ls[-1]]

def build(src, dst, execute=True):
    core = open(os.path.join(os.path.dirname(os.path.abspath(src)), '..', 'egt_core.py')).read()
    assert "'''" not in core
    cells = [(k, t.replace('@@EGT_CORE@@', core)) for k, t in parse(src)]; nb_cells = []; ns = {"__name__": "__main__"}; count = 0
    pending = []
    def fake_show(*a, **k):
        for num in plt.get_fignums():
            f = plt.figure(num); buf = io.BytesIO(); f.savefig(buf, format="png", dpi=80, bbox_inches="tight")
            pending.append(base64.b64encode(buf.getvalue()).decode()); plt.close(f)
    plt.show = fake_show
    for kind, text in cells:
        if kind == "markdown":
            nb_cells.append({"cell_type": "markdown", "metadata": {}, "source": to_source(text)}); continue
        count += 1; outputs = []
        if execute:
            buf = io.StringIO(); t0 = time.time()
            try:
                with contextlib.redirect_stdout(buf):
                    exec(compile(text, f"<cell {count}>", "exec"), ns)
            except Exception:
                tb = traceback.format_exc(); print(tb, file=sys.stderr)
                outputs.append({"output_type": "error", "ename": "Error", "evalue": tb.splitlines()[-1], "traceback": tb.splitlines()})
            print(f"cell {count}: {time.time()-t0:.1f}s", file=sys.stderr)
            if buf.getvalue():
                outputs.insert(0, {"output_type": "stream", "name": "stdout", "text": to_source(buf.getvalue().rstrip("\n") + "\n")})
            for img in pending:
                outputs.append({"output_type": "display_data", "data": {"image/png": img, "text/plain": ["<Figure>"]}, "metadata": {}})
            pending.clear()
        nb_cells.append({"cell_type": "code", "execution_count": count if execute else None, "metadata": {},
                         "outputs": outputs, "source": to_source(text)})
    nb = {"cells": nb_cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
          "language_info": {"name": "python", "version": "3.12"}}, "nbformat": 4, "nbformat_minor": 5}
    for i, c in enumerate(nb["cells"]): c["id"] = f"cell-{i:02d}"
    json.dump(nb, open(dst, "w"), indent=1)

if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2], execute=(len(sys.argv) < 4 or sys.argv[3] != "noexec"))
    # copy figures into paper/figures (the paper folder has to be self-contained for Overleaf)
    import shutil, glob
    os.makedirs("paper/figures", exist_ok=True)
    for f in glob.glob("figures/*.png"):
        shutil.copy2(f, os.path.join("paper/figures", os.path.basename(f)))
