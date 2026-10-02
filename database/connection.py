"""MySQL access + named SQL templates loaded from database/queries.sql.

Templates contain two kinds of placeholders:
  {identifier}   - table / column / SQL expression, filled ONLY from the whitelisted dataset registry
  %(param)s      - values, passed to the driver as bound parameters (never string-formatted)
"""
from __future__ import annotations

import pathlib
import re
from contextlib import contextmanager

import pandas as pd
import pymysql

import config

_SQL_FILE = pathlib.Path(__file__).with_name("queries.sql")
_TEMPLATES: dict[str, str] = {}


def templates() -> dict[str, str]:
    if not _TEMPLATES:
        name, buf = None, []
        for line in _SQL_FILE.read_text(encoding="utf-8").splitlines():
            m = re.match(r"--\s*name:\s*(\w+)", line)
            if m:
                if name:
                    _TEMPLATES[name] = "\n".join(buf).strip().rstrip(";")
                name, buf = m.group(1), []
            elif name:
                buf.append(line)
        if name:
            _TEMPLATES[name] = "\n".join(buf).strip().rstrip(";")
    return _TEMPLATES


@contextmanager
def connect(database: str | None = None):
    if not config.DB["password"]:
        raise RuntimeError("Set ADLENS_DB_PASSWORD (and ADLENS_DB_USER if not 'analytics') for the local MySQL server.")
    conn = pymysql.connect(**config.DB, database=database or config.WORK_DB, autocommit=True, local_infile=True)
    try:
        yield conn
    finally:
        conn.close()


def query(sql: str, params: dict | None = None, database: str | None = None) -> pd.DataFrame:
    with connect(database) as conn, conn.cursor() as cur:
        cur.execute(sql, params or {})
        cols = [d[0] for d in cur.description] if cur.description else []
        return pd.DataFrame(cur.fetchall(), columns=cols)


def run_template(name: str, identifiers: dict, params: dict | None = None) -> pd.DataFrame:
    sql = templates()[name].format(**identifiers)
    return query(sql, params)


def execute(sql: str, params=None, many: bool = False, database: str | None = None) -> int:
    with connect(database) as conn, conn.cursor() as cur:
        if many:
            return cur.executemany(sql, params)
        return cur.execute(sql, params)


def execute_script(path: pathlib.Path, database: str | None = None) -> None:
    statements = [s.strip() for s in re.sub(r"--.*", "", path.read_text(encoding="utf-8")).split(";") if s.strip()]
    for s in statements:
        execute(s, database=database)
