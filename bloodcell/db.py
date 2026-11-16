"""PostgreSQL persistence for analyses."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import psycopg2
from psycopg2.extras import Json, RealDictCursor

from bloodcell.config import PROJECT_ROOT, get_settings
from bloodcell.schemas import Analysis

SCHEMA_FILE = PROJECT_ROOT / "db" / "init.sql"


@contextmanager
def connect(database_url: str | None = None) -> Iterator[psycopg2.extensions.connection]:
    url = database_url or get_settings().database_url
    if not url:
        raise RuntimeError("DATABASE_URL is not set")
    conn = psycopg2.connect(url)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(database_url: str | None = None) -> None:
    with connect(database_url) as conn, conn.cursor() as cur:
        cur.execute(SCHEMA_FILE.read_text())


def save_analysis(
    analysis: Analysis, image_jpeg: bytes | None = None, database_url: str | None = None
) -> int:
    counts = analysis.counts
    with connect(database_url) as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO blood_cell_detections (
                filename, image_width, image_height, model_name, inference_ms,
                total_detections, rbc_count, wbc_count, platelet_count,
                qc_status, quality_control, detections, ai_report, report_source,
                image_jpeg
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            RETURNING id
            """,
            (
                analysis.filename,
                analysis.image_width,
                analysis.image_height,
                analysis.model_name,
                analysis.inference_ms,
                analysis.total_detections,
                counts.get("RBC", 0),
                counts.get("WBC", 0),
                counts.get("Platelets", 0),
                analysis.quality_control.status,
                Json(analysis.quality_control.model_dump()),
                Json([d.model_dump(by_alias=True) for d in analysis.detections]),
                analysis.ai_report,
                analysis.report_source,
                psycopg2.Binary(image_jpeg) if image_jpeg else None,
            ),
        )
        return int(cur.fetchone()[0])


def list_analyses(limit: int = 50, database_url: str | None = None) -> list[dict]:
    with connect(database_url) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT id, filename, model_name, total_detections, rbc_count,
                   wbc_count, platelet_count, qc_status, inference_ms,
                   report_source, created_at
            FROM blood_cell_detections
            ORDER BY created_at DESC, id DESC
            LIMIT %s
            """,
            (limit,),
        )
        return [dict(row) for row in cur.fetchall()]


def get_analysis(analysis_id: int, database_url: str | None = None) -> dict | None:
    with connect(database_url) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            "SELECT * FROM blood_cell_detections WHERE id = %s", (analysis_id,)
        )
        row = cur.fetchone()
        if not row:
            return None
        row = dict(row)
        row["has_image"] = row.pop("image_jpeg", None) is not None
        return row


def encode_jpeg(image, quality: int = 90) -> bytes:
    """Compress a PIL image for storage (~60-80 kB for a BCCD frame)."""
    import io

    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()
