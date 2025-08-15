from __future__ import annotations
import os, json, uuid, shutil, gzip
from datetime import datetime, timezone

from airflow import DAG
from airflow.decorators import task
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from docker.types import Mount


import boto3
import psycopg
from psycopg.rows import dict_row

# ---------- minimal config ----------
RESULTS_DSN = os.environ["RESULTS_DSN"]                         
S3_BUCKET = os.environ["S3_BUCKET"]                             
S3_PREFIX = os.environ.get("S3_PREFIX", "").strip("/")          
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4.1-mini")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
S3_ENDPOINT_URL = os.environ.get("S3_ENDPOINT_URL") or None

LOCAL_BASE = "/home/airflow/.exec"                       

def s3():
    return boto3.client("s3", region_name=AWS_REGION, endpoint_url=S3_ENDPOINT_URL or None)

def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def pg_dsn(dsn: str) -> str:
    return dsn.replace("postgresql+psycopg://", "postgresql://")

# ---------- DAG ----------
with DAG(
    dag_id="doc_classification_pipeline",
    start_date=days_ago(1),
    schedule=None,           
    catchup=False,
    tags=["ml", "docs"],
    default_args={"owner": "you", "retries": 0},
) as dag:

    @task
    def start_run() -> dict:
        run_id = f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        work_dir = os.path.join(LOCAL_BASE, run_id)
        raw_dir  = os.path.join(work_dir, "raw")
        llm_dir  = os.path.join(work_dir, "llm")
        out_dir  = os.path.join(work_dir, "out")
        for d in (raw_dir, llm_dir, out_dir):
            os.makedirs(d, exist_ok=True)


        with psycopg.connect(pg_dsn(RESULTS_DSN), row_factory=dict_row) as conn:
            conn.execute("""
                INSERT INTO runs (run_id, dag_id, ts_started, status, model_version, data_snapshot_uri)
                VALUES (%s, %s, NOW(), %s, %s, %s)
            """, (run_id, dag.dag_id, "running", MODEL_NAME, f"s3://{S3_BUCKET}/{S3_PREFIX}"))
            conn.commit()

        return {
            "run_id": run_id,
            "work_dir": work_dir,
            "raw_dir": raw_dir,
            "llm_dir": llm_dir,
            "out_dir": out_dir,
        }

    @task
    def download_pdfs(ctx: dict) -> dict:
        """
        List and download all PDFs under s3://S3_BUCKET/S3_PREFIX
        Keep it simple: flat listing; filter *.pdf
        """
        prefix = S3_PREFIX
        cli = s3()
        paginator = cli.get_paginator("list_objects_v2")
        n = 0
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not key.lower().endswith(".pdf"):
                    continue
                local_path = os.path.join(ctx["raw_dir"], os.path.basename(key))
                cli.download_file(S3_BUCKET, key, local_path)
                n += 1
        if n == 0:
            raise RuntimeError(f"No PDFs under s3://{S3_BUCKET}/{prefix}")
        return ctx

    extract = DockerOperator(
        task_id="extract_pdfs",
        image="docproc:latest",
        api_version="auto",
        auto_remove=True,
        command=[
            "run_extractor_batch",
            "--in-dir", "/.exec/{{ ti.xcom_pull(task_ids='start_run')['run_id'] }}/raw",
            "--outdir", "/.exec/{{ ti.xcom_pull(task_ids='start_run')['run_id'] }}/llm",
        ],
        environment={},  
        mount_tmp_dir=False,
        mounts=[Mount(source="ashvinai_exec", target="/.exec", type="volume")],
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
    )

    classify = DockerOperator(
        task_id="classify_llm",
        image="docproc:latest",
        api_version="auto",
        auto_remove=True,
        command=[
            "classification",
            "/.exec/{{ ti.xcom_pull(task_ids='start_run')['run_id'] }}/llm",
            "--outdir", "/.exec/{{ ti.xcom_pull(task_ids='start_run')['run_id'] }}/out",
            "--model", MODEL_NAME,
            "--max", "0"
        ],
        environment={"OPENAI_API_KEY": OPENAI_API_KEY},
        mount_tmp_dir=False,
        mounts=[Mount(source="ashvinai_exec", target="/.exec", type="volume")],
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
    )

    @task
    def upload_artifacts(ctx: dict) -> dict:
        """
        Upload classifier outputs to:
          s3://<bucket>/<S3_PREFIX>/../classifications/run=<run_id>/
        (we reuse the same root, keeping it simple)
        """
        run_id = ctx["run_id"]
        out_dir = ctx["out_dir"]
        jpath = os.path.join(out_dir, "classifications.jsonl")
        cpath = os.path.join(out_dir, "classifications.csv")
        if not os.path.isfile(jpath):
            raise RuntimeError("classifications.jsonl not found")

        gz_path = jpath + ".gz"
        with open(jpath, "rb") as f_in, gzip.open(gz_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

        pred_prefix = "/".join([p for p in [S3_PREFIX.rsplit("/", 1)[0] if "/" in S3_PREFIX else "", "classifications", f"run={run_id}"] if p])
        cli = s3()
        cli.upload_file(gz_path, S3_BUCKET, f"{pred_prefix}/classifications.jsonl.gz")
        if os.path.isfile(cpath):
            cli.upload_file(cpath, S3_BUCKET, f"{pred_prefix}/classifications.csv")

        ctx["s3_classifications_uri"] = f"s3://{S3_BUCKET}/{pred_prefix}/classifications.jsonl.gz"
        return ctx

    @task
    def load_to_db(ctx: dict) -> dict:
        """Upsert JSONL into Postgres with run + model annotations."""
        run_id = ctx["run_id"]
        jpath = os.path.join(ctx["out_dir"], "classifications.jsonl")

        rows = []
        with open(jpath, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                rows.append((
                    rec.get("doc_id"),
                    rec.get("source_file"),
                    rec.get("predicted_type"),
                    rec.get("dated") or "",
                    bool(rec.get("needs_human_review", False)),
                    json.dumps(rec.get("key_facts", [])), 
                    (rec.get("confidence_reasoning", "") or "")[:4000],
                    MODEL_NAME,
                    run_id,
                ))
        if not rows:
            raise RuntimeError("No rows to insert")

        with psycopg.connect(pg_dsn(RESULTS_DSN), autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.executemany("""
                    INSERT INTO classifications
                    (doc_id, source_file, predicted_type, dated, needs_human_review,
                    key_facts, confidence_reasoning, model_version, run_id)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (doc_id, model_version)
                    DO UPDATE SET
                    predicted_type = EXCLUDED.predicted_type,
                    dated = EXCLUDED.dated,
                    needs_human_review = EXCLUDED.needs_human_review,
                    key_facts = EXCLUDED.key_facts,
                    confidence_reasoning = EXCLUDED.confidence_reasoning,
                    run_id = EXCLUDED.run_id,
                    created_at = NOW();
                """, rows)

                cur.execute("""
                    UPDATE runs SET ts_finished=NOW(), status='success', preds_uri=%s
                    WHERE run_id=%s
                """, (ctx.get("s3_classifications_uri"), run_id))           
        return ctx        

    visualize = DockerOperator(
        task_id="visualize_results",
        image="streamlit-viz:latest",  
        api_version="auto",
        auto_remove=True,
        command=[
            "timeout", "300", 
            "streamlit", "run", "/app/visuals.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--server.headless", "true"
        ],
        environment={
            "RESULTS_DSN": RESULTS_DSN,
            "RUN_ID": "{{ ti.xcom_pull(task_ids='start_run')['run_id'] }}",
            "MODEL_NAME": MODEL_NAME
        },
        mount_tmp_dir=False,
        mounts=[
            Mount(source="ashvinai_exec", target="/home/streamlit/.exec", type="volume", read_only=True)
        ],
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",  
    )

    ctx0 = start_run()
    ctx1 = download_pdfs(ctx0)
    e = extract; e.set_upstream(ctx1)
    c = classify; c.set_upstream(e)
    ctx2 = upload_artifacts(ctx0); ctx2.set_upstream(c)
    ctx3 = load_to_db(ctx2)
    v = visualize; v.set_upstream(ctx3)
