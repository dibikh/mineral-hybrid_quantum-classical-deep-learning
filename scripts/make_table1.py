import argparse, os
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="outputs/predictions_test.csv")
    ap.add_argument("--out_csv", default="outputs/table1_metrics.csv")
    ap.add_argument("--out_tex", default="outputs/table1_metrics.tex")
    args = ap.parse_args()

    m = pd.read_csv(args.out_csv) if os.path.exists(args.out_csv) else None
    if m is None:
        raise FileNotFoundError(f"Missing {args.out_csv}. Run `python src/inference.py` first.")

    # Produce LaTeX table
    tex = m.to_latex(index=False, float_format="%.4f", caption="Test-set performance comparison.", label="tab:test_metrics")
    os.makedirs(os.path.dirname(args.out_tex), exist_ok=True)
    with open(args.out_tex, "w", encoding="utf-8") as f:
        f.write(tex)
    print(f"[OK] Wrote {args.out_tex}")

if __name__ == "__main__":
    main()
