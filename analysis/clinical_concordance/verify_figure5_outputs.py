"""Independently verify Figure 5's R outputs against the frozen release."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t, ttest_rel
from build_mean_rank_percentile_source import DEFAULT_INPUT, DEFAULT_OUTPUT, METHOD_COLUMNS, METHOD_NAMES

METHODS = list(METHOD_NAMES.values())


def near(actual, expected):
    np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=1e-9)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir
    frozen = pd.read_csv(DEFAULT_INPUT / "validated_gene_star_motifs_1630_mean10.tsv", sep="\t")
    values = frozen[list(METHOD_COLUMNS.values())].to_numpy()
    clusters, ids = np.unique(frozen["disease_mesh_id"] + ":" + frozen["drugbank_id"], return_inverse=True)
    n, g = len(values), len(clusters)
    assert (n, g) == (1630, 785)
    best = values == values.min(axis=1, keepdims=True)
    assert (best.sum(axis=1) == 1).all()
    near(best.sum(axis=0), [334, 145, 215, 412, 524])

    def cr1(x):
        mean = np.mean(x)
        scores = np.bincount(ids, weights=x-mean)
        se = np.sqrt(g / (g-1) * np.sum(scores**2) / n**2)
        return mean, se

    bars = pd.read_csv(output / "figure5_plot_source.csv")
    assert len(bars) == 35
    assert list(bars["panel"].drop_duplicates()) == list("abcdefg")
    for panel, matrix in (("a", best * 100.0), ("b", values)):
        section = bars[bars.panel == panel]
        assert section.method.tolist() == METHODS
        for j, (_, row) in enumerate(section.iterrows()):
            mean, se = cr1(matrix[:, j])
            halfwidth = t.ppf(.975, g-1) * se
            near([row.value, row.error_lower, row.error_upper, row.error_se],
                 [mean, mean-halfwidth, mean+halfwidth, se])
            assert row.error_n == g and "95%" in row.error_type
    near(bars[bars.panel == "b"].value, [43.2080119646611, 44.9114317003886,
                                      39.0275290065927, 36.577397109483, 35.6925691261184])
    tests = pd.read_csv(output / "figure5_complete_cohort_cluster_robust_results.csv").set_index("panel")
    for panel, differences, upper in (("a", best[:,4].astype(float)-best[:,3], True),
                                      ("b", values[:,4]-values[:,3], False)):
        mean, se = cr1(differences)
        p = t.sf(mean/se, g-1) if upper else t.cdf(mean/se, g-1)
        row = tests.loc[panel]
        near([row.mean_difference, row.cluster_robust_se, row.p_raw], [mean,se,p])
        assert row.df == 784 and row.baseline == "TriMoGCL" and row.significance == "*"
    near(tests.p_raw, [.0151253264000526, .0199482574390034])

    seeds = pd.read_csv(output / "figure5_case_seed_percentiles.csv")
    case_tests = pd.read_csv(output / "figure5_case_seed_tests.csv").set_index("panel")
    case_ids = [3999, 14585, 40511, 6118, 35674]
    expected_p = [.0101861403208684, .026368750391558, .0254077860812336,
                  .0345670177312038, .0253991974414998]
    for panel, candidate, p_expected in zip("cdefg", case_ids, expected_p):
        source = seeds[seeds.panel == panel]
        assert set(source.candidate_index) == {candidate}
        wide = source.pivot(index="seed", columns="method", values="rank_percentile")[METHODS]
        assert wide.index.tolist() == list(range(0,100,10))
        means, sds = wide.mean(), wide.std(ddof=1)
        comparator = means.drop("HANAMI").idxmin()
        test = ttest_rel(wide["HANAMI"], wide[comparator], alternative="less")
        row = case_tests.loc[panel]
        assert row.comparison_method == comparator and row.df == 9 and row.n == 10
        near([row.p_value, row.p_value], [test.pvalue, p_expected])
        section = bars[bars.panel == panel].set_index("method").loc[METHODS]
        near(section.value, means)
        near(section.error_lower, means-sds)
        near(section.error_upper, means+sds)
        assert set(section.significance) == {"*"}
        assert set(section.error_type) == {"Sample SD across ten seeds"}
    assert (output / "figure5.pdf").read_bytes().startswith(b"%PDF-")
    print("PASS: 35 bars, seven tests, CR1 intervals, ten-seed SDs and current case identities.")


if __name__ == "__main__":
    main()
