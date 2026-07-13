"""HTML report builder for NLA bias visualization."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from src.main.nla.visualization import (
    find_condition_contrast_groups,
    make_condition_contrast_html,
    make_keyword_table,
    make_token_timeline_html,
    write_aggregate_figures,
)

logger = logging.getLogger(__name__)

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"

_TABLE_COLUMNS = [
    "example_id",
    "marcador_codigo",
    "condition_gender",
    "condition_race",
    "disciplina_codigo",
    "token_str",
    "token_index",
    "token_role",
    "nla_layer",
    "nla_explanation",
    "reconstruction_mse",
    "reconstruction_cosine",
    "areas_recomendadas",
    "sexo_atribuido",
    "cor_ou_raca",
    "nome",
    "idade",
    "estado",
    "renda_mensal",
]

_FIGURE_LABELS = {
    "mse_by_marcador.png": "MSE by marcador",
    "cosine_by_marcador.png": "Cosine by marcador",
    "mse_by_gender.png": "MSE by gender",
    "cosine_by_gender.png": "Cosine by gender",
    "mse_by_race.png": "MSE by race",
    "cosine_by_race.png": "Cosine by race",
    "mse_by_token_role.png": "MSE by token role",
    "cosine_by_token_role.png": "Cosine by token role",
    "mse_by_layer.png": "MSE by layer",
    "cosine_by_layer.png": "Cosine by layer",
    "mse_by_token_position.png": "MSE by token position",
}


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
    )


def _load_css() -> str:
    return (_TEMPLATE_DIR / "report.css").read_text(encoding="utf-8")


def _format_cell(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    text = str(value)
    if len(text) > 240:
        return text[:237] + "..."
    return text


def _inspection_rows(df: pd.DataFrame, max_rows: int) -> list[dict]:
    cols = [c for c in _TABLE_COLUMNS if c in df.columns]
    work = df[cols].copy()
    if "reconstruction_mse" in work.columns:
        work = work.sort_values("reconstruction_mse", ascending=False)
    work = work.head(max_rows)
    rows = []
    for _, r in work.iterrows():
        rows.append({c: _format_cell(r.get(c)) for c in cols})
    return rows


def build_html_report(
    merged_df: pd.DataFrame,
    out_dir: str | Path,
    run_id: str,
    max_table_rows: int = 2000,
    max_sample_pages: int | None = None,
) -> Path:
    """Write index.html, figures, keyword tables, sample and contrast pages."""
    out_dir = Path(out_dir)
    figures_dir = out_dir / "figures"
    tables_dir = out_dir / "tables"
    samples_dir = out_dir / "samples"
    contrasts_dir = out_dir / "contrasts"
    for d in (out_dir, figures_dir, tables_dir, samples_dir, contrasts_dir):
        d.mkdir(parents=True, exist_ok=True)

    df = merged_df.copy()
    if "condition_gender" not in df.columns and "marcador_codigo" in df.columns:
        from src.main.nla.analysis import enrich_condition_columns

        df = enrich_condition_columns(df)

    logger.info("writing aggregate figures")
    written_figs = write_aggregate_figures(df, figures_dir)
    figures = []
    for path in written_figs:
        name = path.name
        figures.append(
            {
                "href": f"figures/{name}",
                "label": _FIGURE_LABELS.get(name, name),
            }
        )

    keyword_tables = []
    for group_col, label in (
        ("condition_gender", "keyword_logodds_gender"),
        ("condition_race", "keyword_logodds_race"),
        ("marcador_codigo", "keyword_logodds_marcador"),
    ):
        if group_col not in df.columns:
            continue
        table = make_keyword_table(df, group_col=group_col, n=20)
        csv_name = f"{label}.csv"
        csv_path = tables_dir / csv_name
        table.to_csv(csv_path, index=False)
        keyword_tables.append({"href": f"tables/{csv_name}", "label": csv_name})

    # Sample pages
    example_ids = df["example_id"].drop_duplicates().tolist()
    if max_sample_pages is not None:
        # Prefer examples with highest MSE spread for qualitative inspection
        spread = (
            df.groupby("example_id")["reconstruction_mse"]
            .agg(lambda s: float(s.max() - s.min()) if len(s) else 0.0)
            .sort_values(ascending=False)
        )
        example_ids = spread.head(max_sample_pages).index.tolist()

    env = _env()
    css = _load_css()
    sample_tmpl = env.get_template("sample.html")
    sample_links = []
    for example_id in example_ids:
        example_df = df[df["example_id"] == example_id]
        body = make_token_timeline_html(example_df)
        page = sample_tmpl.render(css=css, example_id=example_id, body_html=body)
        fname = f"{example_id}.html"
        (samples_dir / fname).write_text(page, encoding="utf-8")
        sample_links.append({"href": f"samples/{fname}", "label": example_id})

    # Contrast pages
    contrast_tmpl = env.get_template("contrast.html")
    contrast_links = []
    for kind in ("gender", "race"):
        for contrast in find_condition_contrast_groups(df, kind=kind):
            body = make_condition_contrast_html(df, contrast)
            key = contrast["contrast_key"]
            page = contrast_tmpl.render(css=css, contrast_key=key, body_html=body)
            fname = f"{key}.html"
            (contrasts_dir / fname).write_text(page, encoding="utf-8")
            disc = contrast["match_keys"].get("disciplina_codigo", "")
            contrast_links.append(
                {
                    "href": f"contrasts/{fname}",
                    "label": f"{kind}: {disc}",
                }
            )

    table_columns = [c for c in _TABLE_COLUMNS if c in df.columns]
    table_rows = _inspection_rows(df, max_table_rows)
    conditions = sorted(df["marcador_codigo"].dropna().astype(str).unique().tolist()) if "marcador_codigo" in df.columns else []

    index_html = env.get_template("index.html").render(
        css=css,
        run_id=run_id,
        n_rows=len(df),
        n_examples=df["example_id"].nunique() if "example_id" in df.columns else 0,
        figures=figures,
        keyword_tables=keyword_tables,
        table_columns=table_columns,
        table_rows=table_rows,
        conditions=conditions,
        sample_links=sample_links,
        contrast_links=contrast_links,
    )
    index_path = out_dir / "index.html"
    index_path.write_text(index_html, encoding="utf-8")
    logger.info("wrote report index to %s", index_path)
    return index_path
