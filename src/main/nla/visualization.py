"""Static figures and HTML fragments for NLA bias visualization."""

from __future__ import annotations

import html
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer

from src.main.nla.analysis import tokenize_explanation

_MATCH_KEYS = ("modelo", "temperatura", "repeticao", "disciplina_codigo")
_GENDER_MARKERS = ("sem_marcador_social", "feminino", "masculino")
_RACE_MARKERS = (
    "sem_marcador_social",
    "branca",
    "preta",
    "parda",
    "amarela",
    "indigena",
)


def _ensure_parent(path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _safe_group_order(series: pd.Series) -> list:
    vals = sorted(series.dropna().astype(str).unique().tolist())
    return vals


def make_mse_boxplot(
    df: pd.DataFrame,
    by: str = "marcador_codigo",
    out_path: str | Path | None = None,
    title: str | None = None,
) -> Path | plt.Figure:
    """Boxplot of reconstruction_mse grouped by ``by``."""
    if by not in df.columns or "reconstruction_mse" not in df.columns:
        raise KeyError(f"missing columns for MSE boxplot: {by}, reconstruction_mse")

    plot_df = df[[by, "reconstruction_mse"]].dropna()
    fig, ax = plt.subplots(figsize=(max(8, 0.55 * plot_df[by].nunique()), 5))
    order = _safe_group_order(plot_df[by])
    sns.boxplot(data=plot_df, x=by, y="reconstruction_mse", order=order, ax=ax)
    ax.set_title(title or f"Reconstruction MSE by {by}")
    ax.set_xlabel(by)
    ax.set_ylabel("reconstruction_mse")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()

    if out_path is None:
        return fig
    path = _ensure_parent(Path(out_path))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def make_cosine_boxplot(
    df: pd.DataFrame,
    by: str = "marcador_codigo",
    out_path: str | Path | None = None,
    title: str | None = None,
) -> Path | plt.Figure:
    """Boxplot of reconstruction_cosine grouped by ``by``."""
    if by not in df.columns or "reconstruction_cosine" not in df.columns:
        raise KeyError(f"missing columns for cosine boxplot: {by}, reconstruction_cosine")

    plot_df = df[[by, "reconstruction_cosine"]].dropna()
    fig, ax = plt.subplots(figsize=(max(8, 0.55 * plot_df[by].nunique()), 5))
    order = _safe_group_order(plot_df[by])
    sns.boxplot(data=plot_df, x=by, y="reconstruction_cosine", order=order, ax=ax)
    ax.set_title(title or f"Reconstruction cosine by {by}")
    ax.set_xlabel(by)
    ax.set_ylabel("reconstruction_cosine")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()

    if out_path is None:
        return fig
    path = _ensure_parent(Path(out_path))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def make_mse_by_token_position(
    df: pd.DataFrame,
    out_path: str | Path | None = None,
    title: str | None = None,
) -> Path | plt.Figure:
    """Mean MSE (± stderr) vs token_index."""
    needed = {"token_index", "reconstruction_mse"}
    if not needed.issubset(df.columns):
        raise KeyError(f"missing columns for position plot: {needed}")

    grouped = (
        df.groupby("token_index", as_index=False)
        .agg(
            mean_mse=("reconstruction_mse", "mean"),
            std_mse=("reconstruction_mse", "std"),
            n=("reconstruction_mse", "count"),
        )
        .sort_values("token_index")
    )
    grouped["stderr"] = grouped["std_mse"] / np.sqrt(grouped["n"].clip(lower=1))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(grouped["token_index"], grouped["mean_mse"], color="#1f4e79", lw=1.5)
    ax.fill_between(
        grouped["token_index"],
        grouped["mean_mse"] - grouped["stderr"].fillna(0),
        grouped["mean_mse"] + grouped["stderr"].fillna(0),
        color="#1f4e79",
        alpha=0.2,
    )
    ax.set_title(title or "Mean reconstruction MSE by token position")
    ax.set_xlabel("token_index")
    ax.set_ylabel("mean reconstruction_mse")
    fig.tight_layout()

    if out_path is None:
        return fig
    path = _ensure_parent(Path(out_path))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _ngrams(tokens: list[str], n: int) -> list[str]:
    if n <= 1:
        return tokens
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _top_ngrams_for_group(texts: Iterable[str], ngram: int, top_n: int) -> list[tuple[str, int]]:
    counts: Counter[str] = Counter()
    for text in texts:
        toks = tokenize_explanation(text)
        counts.update(_ngrams(toks, ngram))
    return counts.most_common(top_n)


def _log_odds_scores(
    group_texts: list[str],
    rest_texts: list[str],
    alpha: float = 0.01,
) -> pd.DataFrame:
    """Smoothed log-odds ratio of unigrams in group vs rest (Monroe-style)."""
    group_counts: Counter[str] = Counter()
    rest_counts: Counter[str] = Counter()
    for text in group_texts:
        group_counts.update(tokenize_explanation(text))
    for text in rest_texts:
        rest_counts.update(tokenize_explanation(text))

    vocab = set(group_counts) | set(rest_counts)
    n_group = sum(group_counts.values())
    n_rest = sum(rest_counts.values())
    rows = []
    for term in vocab:
        y_i = group_counts[term]
        y_j = rest_counts[term]
        # prior from combined counts
        prior = group_counts[term] + rest_counts[term]
        a = alpha * (prior + 1)
        p_i = (y_i + a) / (n_group + a * len(vocab))
        p_j = (y_j + a) / (n_rest + a * len(vocab) if n_rest else 1.0)
        log_odds = math.log(p_i) - math.log(p_j)
        # approximate variance for z-score
        var = 1.0 / (y_i + a) + 1.0 / (y_j + a)
        z = log_odds / math.sqrt(var) if var > 0 else 0.0
        rows.append(
            {
                "term": term,
                "count_group": y_i,
                "count_rest": y_j,
                "log_odds": log_odds,
                "z_score": z,
            }
        )
    return pd.DataFrame(rows)


def make_keyword_table(
    df: pd.DataFrame,
    group_col: str,
    n: int = 20,
    text_col: str = "nla_explanation",
) -> pd.DataFrame:
    """Top n-grams, TF-IDF terms, and log-odds keywords per condition group."""
    if group_col not in df.columns or text_col not in df.columns:
        raise KeyError(f"missing columns for keyword table: {group_col}, {text_col}")

    work = df[[group_col, text_col]].dropna()
    work[text_col] = work[text_col].astype(str)
    groups = _safe_group_order(work[group_col])
    rows: list[dict] = []

    # TF-IDF over concatenated docs per group
    docs = []
    for g in groups:
        docs.append(" ".join(work.loc[work[group_col] == g, text_col].tolist()))
    tfidf_terms: dict[str, list[tuple[str, float]]] = {g: [] for g in groups}
    if docs and any(d.strip() for d in docs):
        vectorizer = TfidfVectorizer(
            max_features=500,
            token_pattern=r"[a-zA-ZÀ-ÿ0-9']+",
            ngram_range=(1, 2),
            min_df=1,
        )
        try:
            matrix = vectorizer.fit_transform(docs)
            terms = np.array(vectorizer.get_feature_names_out())
            for i, g in enumerate(groups):
                scores = matrix[i].toarray().ravel()
                top_idx = scores.argsort()[::-1][:n]
                tfidf_terms[g] = [
                    (terms[j], float(scores[j])) for j in top_idx if scores[j] > 0
                ]
        except ValueError:
            pass

    for g in groups:
        group_texts = work.loc[work[group_col] == g, text_col].tolist()
        rest_texts = work.loc[work[group_col] != g, text_col].tolist()

        for rank, (term, count) in enumerate(
            _top_ngrams_for_group(group_texts, ngram=1, top_n=n), start=1
        ):
            rows.append(
                {
                    "group": g,
                    "method": "top_unigram",
                    "rank": rank,
                    "term": term,
                    "score": float(count),
                }
            )
        for rank, (term, count) in enumerate(
            _top_ngrams_for_group(group_texts, ngram=2, top_n=n), start=1
        ):
            rows.append(
                {
                    "group": g,
                    "method": "top_bigram",
                    "rank": rank,
                    "term": term,
                    "score": float(count),
                }
            )
        for rank, (term, score) in enumerate(tfidf_terms.get(g, []), start=1):
            rows.append(
                {
                    "group": g,
                    "method": "tfidf",
                    "rank": rank,
                    "term": term,
                    "score": score,
                }
            )

        if rest_texts:
            lodds = _log_odds_scores(group_texts, rest_texts)
            if not lodds.empty:
                top = lodds.sort_values("z_score", ascending=False).head(n)
                for rank, (_, r) in enumerate(top.iterrows(), start=1):
                    rows.append(
                        {
                            "group": g,
                            "method": "log_odds",
                            "rank": rank,
                            "term": r["term"],
                            "score": float(r["z_score"]),
                        }
                    )

    return pd.DataFrame(rows)


def make_token_timeline_html(example_df: pd.DataFrame) -> str:
    """Render ordered tokens with attached NLA explanation metadata."""
    if example_df.empty:
        return "<p>No tokens for this example.</p>"

    ordered = example_df.sort_values("token_index")
    meta = ordered.iloc[0]
    prompt = html.escape(str(meta.get("prompt") or ""))
    output = html.escape(str(meta.get("model_output") or meta.get("resposta_raw") or ""))
    areas = str(meta.get("areas_recomendadas") or "").strip()
    if areas:
        outcome_html = f"<p><strong>areas:</strong> {html.escape(areas)}</p>"
    else:
        profile_bits = []
        for key, label in (
            ("sexo_atribuido", "sexo"),
            ("cor_ou_raca", "cor/raça"),
            ("nome", "nome"),
            ("idade", "idade"),
            ("estado", "estado"),
            ("renda_mensal", "renda"),
        ):
            value = meta.get(key)
            if value is None or (isinstance(value, float) and pd.isna(value)):
                continue
            text = str(value).strip()
            if text:
                profile_bits.append(
                    f"<strong>{label}:</strong> {html.escape(text)}"
                )
        outcome_html = (
            f"<p>{' | '.join(profile_bits)}</p>" if profile_bits else ""
        )

    chips = []
    details = []
    for _, row in ordered.iterrows():
        tid = f"tok-{int(row['token_index'])}"
        token = html.escape(str(row.get("token_str", "")))
        role = html.escape(str(row.get("token_role", "")))
        mse = row.get("reconstruction_mse")
        cos = row.get("reconstruction_cosine")
        expl = html.escape(str(row.get("nla_explanation") or ""))
        mse_s = f"{float(mse):.4f}" if pd.notna(mse) else "n/a"
        cos_s = f"{float(cos):.4f}" if pd.notna(cos) else "n/a"
        title = f"role={role} mse={mse_s} cos={cos_s}"
        chips.append(
            f'<a class="token role-{role}" href="#{tid}" title="{html.escape(title)}">'
            f"{token}</a>"
        )
        details.append(
            f'<section class="token-detail" id="{tid}">'
            f"<h3>Token: <code>{token}</code> "
            f"(index {int(row['token_index'])})</h3>"
            f"<ul>"
            f"<li><strong>role:</strong> {role}</li>"
            f"<li><strong>layer:</strong> {html.escape(str(row.get('nla_layer', '')))}</li>"
            f"<li><strong>MSE:</strong> {mse_s}</li>"
            f"<li><strong>cosine:</strong> {cos_s}</li>"
            f"</ul>"
            f"<p class=\"explanation\">{expl}</p>"
            f"</section>"
        )

    return (
        f'<div class="timeline">'
        f"<h2>Example {html.escape(str(meta.get('example_id', '')))}</h2>"
        f"<p><strong>marcador:</strong> {html.escape(str(meta.get('marcador_codigo', '')))} "
        f"| <strong>disciplina:</strong> {html.escape(str(meta.get('disciplina_codigo', '')))} "
        f"| <strong>gender:</strong> {html.escape(str(meta.get('condition_gender', '')))} "
        f"| <strong>race:</strong> {html.escape(str(meta.get('condition_race', '')))}</p>"
        f'<div class="prompt-block"><h3>Prompt</h3><pre>{prompt}</pre></div>'
        f'<div class="output-block"><h3>Model output</h3><pre>{output}</pre>'
        f"{outcome_html}</div>"
        f'<div class="token-strip"><h3>Tokens</h3><p class="tokens">'
        + " ".join(chips)
        + "</p></div>"
        + '<div class="token-details">'
        + "".join(details)
        + "</div></div>"
    )


def _example_summary_panel(example_df: pd.DataFrame) -> str:
    if example_df.empty:
        return "<p>missing</p>"
    ordered = example_df.sort_values("token_index")
    meta = ordered.iloc[0]
    mse_mean = ordered["reconstruction_mse"].mean()
    cos_mean = ordered["reconstruction_cosine"].mean()
    sample_expl = ""
    markers = ordered[ordered["token_role"] == "demographic_marker"]
    if not markers.empty:
        sample_expl = str(markers.iloc[0].get("nla_explanation") or "")
    elif not ordered.empty:
        sample_expl = str(ordered.iloc[0].get("nla_explanation") or "")

    top_terms = [
        t for t, _ in _top_ngrams_for_group(ordered["nla_explanation"].dropna().astype(str), 1, 8)
    ]
    return (
        f"<h3>{html.escape(str(meta.get('marcador_codigo', '')))}</h3>"
        f"<p><strong>gender:</strong> {html.escape(str(meta.get('condition_gender', '')))}<br>"
        f"<strong>race:</strong> {html.escape(str(meta.get('condition_race', '')))}</p>"
        f"<p><strong>mean MSE:</strong> {mse_mean:.4f}<br>"
        f"<strong>mean cosine:</strong> {cos_mean:.4f}</p>"
        f"<p><strong>prompt:</strong></p>"
        f"<pre>{html.escape(str(meta.get('prompt') or ''))}</pre>"
        f"<p><strong>output:</strong></p>"
        f"<pre>{html.escape(str(meta.get('model_output') or meta.get('resposta_raw') or ''))}</pre>"
        f"<p><strong>sample explanation:</strong></p>"
        f"<p class=\"explanation\">{html.escape(sample_expl)}</p>"
        f"<p><strong>top terms:</strong> {html.escape(', '.join(top_terms))}</p>"
    )


def find_condition_contrast_groups(
    df: pd.DataFrame,
    kind: str = "gender",
) -> list[dict]:
    """Find matched example sets that share interest/model keys and vary markers."""
    markers = _GENDER_MARKERS if kind == "gender" else _RACE_MARKERS
    required = set(_MATCH_KEYS) | {"example_id", "marcador_codigo"}
    if not required.issubset(df.columns):
        return []

    examples = (
        df[list(required)]
        .drop_duplicates(subset=["example_id"])
        .copy()
    )
    groups: list[dict] = []
    for keys, gdf in examples.groupby(list(_MATCH_KEYS), dropna=False):
        key_map = dict(zip(_MATCH_KEYS, keys if isinstance(keys, tuple) else (keys,)))
        present = {
            row["marcador_codigo"]: row["example_id"]
            for _, row in gdf.iterrows()
            if row["marcador_codigo"] in markers
        }
        if len(present) < 2:
            continue
        ordered_markers = [m for m in markers if m in present]
        groups.append(
            {
                "kind": kind,
                "match_keys": key_map,
                "markers": ordered_markers,
                "example_ids": [present[m] for m in ordered_markers],
                "contrast_key": _contrast_key(kind, key_map),
            }
        )
    return groups


def _contrast_key(kind: str, match_keys: dict) -> str:
    disc = str(match_keys.get("disciplina_codigo", "na"))
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", disc)[:80]
    temp = match_keys.get("temperatura", "")
    rep = match_keys.get("repeticao", "")
    return f"{kind}__{safe}__t{temp}_r{rep}"


def make_condition_contrast_html(
    df: pd.DataFrame,
    contrast: dict,
) -> str:
    """Side-by-side panels for one matched demographic contrast set."""
    match_keys = contrast["match_keys"]
    header = (
        f"<h2>Condition contrast ({html.escape(contrast['kind'])})</h2>"
        f"<p><strong>disciplina:</strong> {html.escape(str(match_keys.get('disciplina_codigo')))} "
        f"| <strong>modelo:</strong> {html.escape(str(match_keys.get('modelo')))} "
        f"| <strong>temp:</strong> {html.escape(str(match_keys.get('temperatura')))} "
        f"| <strong>rep:</strong> {html.escape(str(match_keys.get('repeticao')))}</p>"
    )
    panels = []
    for example_id in contrast["example_ids"]:
        example_df = df[df["example_id"] == example_id]
        panels.append(f'<div class="contrast-panel">{_example_summary_panel(example_df)}</div>')

    return (
        f'<div class="contrast">{header}'
        f'<div class="contrast-grid">{"".join(panels)}</div></div>'
    )


def write_aggregate_figures(df: pd.DataFrame, figures_dir: str | Path) -> list[Path]:
    """Write the standard reconstruction-quality figure set."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    groupings = [
        ("marcador_codigo", "mse_by_marcador.png", "cosine_by_marcador.png"),
        ("condition_gender", "mse_by_gender.png", "cosine_by_gender.png"),
        ("condition_race", "mse_by_race.png", "cosine_by_race.png"),
        ("token_role", "mse_by_token_role.png", "cosine_by_token_role.png"),
    ]
    if "nla_layer" in df.columns and df["nla_layer"].nunique() > 1:
        groupings.append(("nla_layer", "mse_by_layer.png", "cosine_by_layer.png"))

    for by, mse_name, cos_name in groupings:
        if by not in df.columns:
            continue
        written.append(make_mse_boxplot(df, by=by, out_path=figures_dir / mse_name))
        written.append(make_cosine_boxplot(df, by=by, out_path=figures_dir / cos_name))

    written.append(
        make_mse_by_token_position(df, out_path=figures_dir / "mse_by_token_position.png")
    )
    return written
