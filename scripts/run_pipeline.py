#!/usr/bin/env python3
"""
run_pipeline.py
================
End-to-end orchestration script for the HDFS Anomaly Detection ML pipeline.
Executes all four stages in sequence:
  1. Data Ingestion & Storage
  2. Feature Engineering
  3. Model Training
  4. Evaluation

Usage:
    python scripts/run_pipeline.py [--stage {all,ingest,features,train,eval}]
                                   [--csv_path PATH]
                                   [--base_dir PATH]

Author : Big Data ML Assignment
"""

import argparse
import logging
import os
import sys
import time
import json

# ── Logging configuration ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("HDFSPipeline")


# ── Helper: create SparkSession ──────────────────────────────────────────────
def get_spark(app_name: str, driver_mem: str = "6g", exec_mem: str = "6g"):
    from pyspark.sql import SparkSession
    spark = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.driver.memory",           driver_mem)
        .config("spark.executor.memory",          exec_mem)
        .config("spark.sql.shuffle.partitions",   "200")
        .config("spark.sql.adaptive.enabled",     "true")
        .config("spark.serializer",
                "org.apache.spark.serializer.KryoSerializer")
        .config("spark.sql.parquet.filterPushdown", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 – Data Ingestion
# ─────────────────────────────────────────────────────────────────────────────
def stage_ingest(base_dir: str, csv_path: str) -> dict:
    import time
    from pyspark.sql import functions as F
    from pyspark.sql.types import (
        StructType, StructField, DoubleType, IntegerType,
    )

    log.info("=== STAGE 1: Data Ingestion ===")
    spark = get_spark("HDFS_Ingest")

    PARQUET_DIR = os.path.join(base_dir, "project", "data", "hdfs_parquet")

    # Build schema
    fields = (
        [StructField(f"E{i}", DoubleType(), True) for i in range(1, 30)]
        + [StructField("Label", IntegerType(), True)]
        + [StructField(c, DoubleType(), True) for c in [
            "total_events", "mean_event_count", "std_event_count",
            "max_event_count", "min_event_count", "nonzero_events",
            "event_sparsity", "event_concentration", "event_balance",
        ]]
    )
    # Add remaining engineered columns from CSV (interaction + polynomial)
    # Use flexible inference for extra cols
    t0 = time.time()
    raw_df = (
        spark.read
        .option("header",      "true")
        .option("inferSchema", "false")
        .option("mode",        "DROPMALFORMED")
        .schema(StructType(fields))
        .csv(csv_path)
    )

    total_rows = raw_df.count()
    log.info(f"  Rows ingested : {total_rows:,}")

    # Validate
    null_counts = raw_df.select(
        [F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in ["E1", "E5", "Label"]]
    ).collect()[0].asDict()
    for col, cnt in null_counts.items():
        status = "OK" if cnt == 0 else f"WARNING: {cnt} nulls"
        log.info(f"  Null check {col}: {status}")

    # Partition & write Parquet
    (
        raw_df.repartition(16, "Label")
        .write.mode("overwrite")
        .option("compression", "snappy")
        .partitionBy("Label")
        .parquet(PARQUET_DIR)
    )
    elapsed = time.time() - t0
    log.info(f"  Parquet written → {PARQUET_DIR}  ({elapsed:.1f}s)")

    spark.stop()
    return {"stage": "ingest", "rows": total_rows, "path": PARQUET_DIR, "time_s": round(elapsed, 1)}


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 – Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────
def stage_features(base_dir: str) -> dict:
    import time
    from pyspark.sql import functions as F
    from pyspark.sql.types import DoubleType
    from pyspark.ml.feature import VectorAssembler, StandardScaler, MinMaxScaler, Imputer
    from pyspark import StorageLevel

    log.info("=== STAGE 2: Feature Engineering ===")
    spark = get_spark("HDFS_Features")

    PARQUET_IN  = os.path.join(base_dir, "project", "data", "hdfs_parquet")
    PARQUET_OUT = os.path.join(base_dir, "project", "data", "features_parquet")

    df = spark.read.parquet(PARQUET_IN).persist(StorageLevel.MEMORY_AND_DISK_SER)
    n  = df.count()
    log.info(f"  Loaded {n:,} rows from Parquet")

    event_cols = [f"E{i}" for i in range(1, 30)]
    stat_cols  = ["total_events", "mean_event_count", "std_event_count",
                  "max_event_count", "min_event_count", "nonzero_events",
                  "event_sparsity", "event_concentration", "event_balance"]

    # Impute
    imputer = Imputer(strategy="median",
                      inputCols=event_cols + stat_cols,
                      outputCols=event_cols + stat_cols)
    df = imputer.fit(df).transform(df)

    # Category aggregates
    cat_map = {
        "cat_data_transfer": ["E2","E5","E9","E11","E21","E23","E25","E26"],
        "cat_error"        : ["E4","E6","E7","E8","E22"],
        "cat_replication"  : ["E3","E15","E16","E19"],
        "cat_admin"        : ["E14","E18","E24","E27","E28","E29"],
    }
    for cat_col, events in cat_map.items():
        df = df.withColumn(cat_col, sum(F.col(e) for e in events))
    df = df.withColumn("error_ratio",
                       F.when(F.col("total_events") > 0,
                              F.col("cat_error") / F.col("total_events")).otherwise(0.0))
    df = (df
          .withColumn("log_total_events",  F.log1p("total_events"))
          .withColumn("has_error",         (F.col("cat_error") > 0).cast(DoubleType()))
          .withColumn("high_sparsity",     (F.col("event_sparsity") > 0.7).cast(DoubleType())))

    feature_cols = (event_cols + stat_cols +
                    list(cat_map.keys()) +
                    ["error_ratio","log_total_events","has_error","high_sparsity"])
    feature_cols = list(dict.fromkeys(feature_cols))

    for c in feature_cols:
        df = df.withColumn(c, F.when(F.col(c).isNull() | F.isnan(c), 0.0).otherwise(F.col(c)))

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features", handleInvalid="skip")
    df = assembler.transform(df)

    std_scaler = StandardScaler(inputCol="raw_features", outputCol="std_features", withMean=True, withStd=True)
    df = std_scaler.fit(df).transform(df)

    mm_scaler = MinMaxScaler(inputCol="raw_features", outputCol="mm_features")
    df = mm_scaler.fit(df).transform(df)

    # Compute class weights
    counts = {r["Label"]: r["count"] for r in df.groupBy("Label").count().collect()}
    n0, n1 = counts.get(0, 1), counts.get(1, 1)
    w0 = (1.0 / n0) * (n0 + n1) / 2.0
    w1 = (1.0 / n1) * (n0 + n1) / 2.0
    df = df.withColumn("class_weight",
                       F.when(F.col("Label") == 0, w0).otherwise(w1))

    # Split
    df = df.withColumn("_rand", F.rand(seed=42))
    train_df = df.filter(F.col("_rand") < 0.6).drop("_rand")
    val_df   = df.filter((F.col("_rand") >= 0.6) & (F.col("_rand") < 0.8)).drop("_rand")
    test_df  = df.filter(F.col("_rand") >= 0.8).drop("_rand")

    write_cols = ["std_features", "mm_features", "raw_features", "Label", "class_weight"]
    t0 = time.time()
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        avail = [c for c in write_cols if c in split_df.columns]
        (split_df.select(avail)
         .write.mode("overwrite").option("compression","snappy")
         .parquet(os.path.join(PARQUET_OUT, split_name)))
        log.info(f"  {split_name} split written")

    elapsed = time.time() - t0
    df.unpersist()
    spark.stop()
    log.info(f"  Feature engineering done in {elapsed:.1f}s")
    return {"stage": "features", "feature_cols": len(feature_cols), "time_s": round(elapsed, 1)}


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 – Model Training
# ─────────────────────────────────────────────────────────────────────────────
def stage_train(base_dir: str) -> dict:
    import time
    from pyspark.sql import functions as F
    from pyspark import StorageLevel
    from pyspark.ml.classification import (
        LogisticRegression, RandomForestClassifier,
        GBTClassifier, LinearSVC,
    )
    from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
    from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

    log.info("=== STAGE 3: Model Training ===")
    spark = get_spark("HDFS_Train")
    spark.sparkContext.setCheckpointDir(os.path.join(base_dir, "project", "data", "checkpoints"))

    FEAT_DIR   = os.path.join(base_dir, "project", "data", "features_parquet")
    MODELS_DIR = os.path.join(base_dir, "project", "data", "models")
    os.makedirs(MODELS_DIR, exist_ok=True)

    train_df = spark.read.parquet(os.path.join(FEAT_DIR, "train")).persist(StorageLevel.MEMORY_AND_DISK_SER)
    val_df   = spark.read.parquet(os.path.join(FEAT_DIR, "val")).persist(StorageLevel.MEMORY_AND_DISK_SER)
    train_df.count(); val_df.count()

    bin_eval  = BinaryClassificationEvaluator(labelCol="Label", rawPredictionCol="rawPrediction")
    mc_f1     = MulticlassClassificationEvaluator(labelCol="Label", predictionCol="prediction", metricName="f1")
    mc_acc    = MulticlassClassificationEvaluator(labelCol="Label", predictionCol="prediction", metricName="accuracy")

    results = {}

    def eval_model(model, name, feat_col, df):
        pred  = model.transform(df)
        return {
            "val_auc": round(bin_eval.evaluate(pred), 4),
            "val_f1" : round(mc_f1.evaluate(pred), 4),
            "val_acc": round(mc_acc.evaluate(pred), 4),
        }

    # Logistic Regression
    log.info("  Training Logistic Regression …")
    t0 = time.time()
    lr_model = LogisticRegression(featuresCol="std_features", labelCol="Label",
                                   maxIter=100, regParam=0.01,
                                   weightCol="class_weight").fit(train_df)
    lr_model.write().overwrite().save(os.path.join(MODELS_DIR, "lr_model"))
    m = eval_model(lr_model, "LR", "std_features", val_df)
    results["LogisticRegression"] = {**m, "train_time_s": round(time.time()-t0, 2)}
    log.info(f"  LR: {m}")

    # Random Forest
    log.info("  Training Random Forest …")
    t0 = time.time()
    rf_model = RandomForestClassifier(featuresCol="raw_features", labelCol="Label",
                                       numTrees=100, maxDepth=10, seed=42).fit(train_df)
    rf_model.write().overwrite().save(os.path.join(MODELS_DIR, "rf_model"))
    m = eval_model(rf_model, "RF", "raw_features", val_df)
    results["RandomForest"] = {**m, "train_time_s": round(time.time()-t0, 2)}
    log.info(f"  RF: {m}")

    # GBT with CrossValidator
    log.info("  Training GBT with CrossValidator …")
    t0 = time.time()
    gbt = GBTClassifier(featuresCol="raw_features", labelCol="Label",
                        maxIter=50, maxDepth=5, stepSize=0.1,
                        checkpointInterval=10, seed=42)
    param_grid = (ParamGridBuilder()
                  .addGrid(gbt.maxDepth, [4, 6])
                  .addGrid(gbt.stepSize, [0.05, 0.1])
                  .build())
    cv = CrossValidator(estimator=gbt, estimatorParamMaps=param_grid,
                        evaluator=BinaryClassificationEvaluator(labelCol="Label"),
                        numFolds=3, parallelism=2, seed=42, collectSubModels=False)
    cv_model = cv.fit(train_df.union(val_df))
    best_gbt  = cv_model.bestModel
    best_gbt.write().overwrite().save(os.path.join(MODELS_DIR, "gbt_model"))
    m = eval_model(best_gbt, "GBT", "raw_features", val_df)
    results["GBT"] = {**m, "train_time_s": round(time.time()-t0, 2),
                      "best_maxDepth": best_gbt.getMaxDepth(),
                      "best_stepSize": best_gbt.getStepSize()}
    log.info(f"  GBT: {m}")

    # Linear SVC
    log.info("  Training Linear SVC …")
    t0 = time.time()
    svc_model = LinearSVC(featuresCol="std_features", labelCol="Label",
                           maxIter=100, regParam=0.01,
                           weightCol="class_weight").fit(train_df)
    svc_model.write().overwrite().save(os.path.join(MODELS_DIR, "svc_model"))
    m = eval_model(svc_model, "SVC", "std_features", val_df)
    results["LinearSVC"] = {**m, "train_time_s": round(time.time()-t0, 2)}
    log.info(f"  SVC: {m}")

    train_df.unpersist(); val_df.unpersist()
    spark.stop()

    with open(os.path.join(base_dir, "project/data/samples/all_results.json"), "w") as fp:
        json.dump({"mllib_models": results}, fp, indent=2)

    return {"stage": "train", "models": list(results.keys()), "results": results}


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 – Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def stage_eval(base_dir: str) -> dict:
    import time
    import numpy as np
    from pyspark import StorageLevel
    from pyspark.ml.classification import (
        LogisticRegressionModel, RandomForestClassificationModel,
        GBTClassificationModel, LinearSVCModel,
    )
    from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

    log.info("=== STAGE 4: Evaluation ===")
    spark = get_spark("HDFS_Eval")

    FEAT_DIR   = os.path.join(base_dir, "project", "data", "features_parquet")
    MODELS_DIR = os.path.join(base_dir, "project", "data", "models")
    OUT_DIR    = os.path.join(base_dir, "project", "data", "samples")
    os.makedirs(OUT_DIR, exist_ok=True)

    models = {
        "Logistic Regression": LogisticRegressionModel.load(os.path.join(MODELS_DIR, "lr_model")),
        "Random Forest"      : RandomForestClassificationModel.load(os.path.join(MODELS_DIR, "rf_model")),
        "GBT"                : GBTClassificationModel.load(os.path.join(MODELS_DIR, "gbt_model")),
        "Linear SVC"         : LinearSVCModel.load(os.path.join(MODELS_DIR, "svc_model")),
    }
    test_df = spark.read.parquet(os.path.join(FEAT_DIR, "test")).persist(StorageLevel.MEMORY_AND_DISK_SER)
    n_test  = test_df.count()
    log.info(f"  Test rows: {n_test:,}")

    bin_eval = BinaryClassificationEvaluator(labelCol="Label", rawPredictionCol="rawPrediction")
    mc_f1    = MulticlassClassificationEvaluator(labelCol="Label", predictionCol="prediction", metricName="f1")
    mc_acc   = MulticlassClassificationEvaluator(labelCol="Label", predictionCol="prediction", metricName="accuracy")

    final_results = {}
    for name, model in models.items():
        pred = model.transform(test_df)
        auc  = bin_eval.evaluate(pred)
        f1   = mc_f1.evaluate(pred)
        acc  = mc_acc.evaluate(pred)
        final_results[name] = {"AUC_ROC": round(auc, 4), "F1": round(f1, 4), "Accuracy": round(acc, 4)}
        log.info(f"  {name:<22}  AUC={auc:.4f}  F1={f1:.4f}  Acc={acc:.4f}")

    best_model = max(final_results, key=lambda k: final_results[k]["F1"])
    log.info(f"  Best model: {best_model} (F1={final_results[best_model]['F1']})")

    with open(os.path.join(OUT_DIR, "final_test_results.json"), "w") as fp:
        json.dump({"stage": "eval", "test_results": final_results, "best_model": best_model}, fp, indent=2)

    test_df.unpersist()
    spark.stop()
    return {"stage": "eval", "results": final_results, "best_model": best_model}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="HDFS Anomaly Detection Pipeline")
    parser.add_argument("--stage",    default="all",
                        choices=["all", "ingest", "features", "train", "eval"],
                        help="Pipeline stage to run")
    parser.add_argument("--csv_path", default="/home/sayan/Lakhveer/HDFS_ML_Dataset.csv",
                        help="Path to raw CSV dataset")
    parser.add_argument("--base_dir", default="/home/sayan/Lakhveer",
                        help="Base project directory")
    args = parser.parse_args()

    pipeline_start = time.time()
    stage_results  = {}

    stages_to_run = {
        "all"     : ["ingest", "features", "train", "eval"],
        "ingest"  : ["ingest"],
        "features": ["features"],
        "train"   : ["train"],
        "eval"    : ["eval"],
    }[args.stage]

    stage_fns = {
        "ingest"  : lambda: stage_ingest(args.base_dir, args.csv_path),
        "features": lambda: stage_features(args.base_dir),
        "train"   : lambda: stage_train(args.base_dir),
        "eval"    : lambda: stage_eval(args.base_dir),
    }

    for stage_name in stages_to_run:
        t0 = time.time()
        log.info(f"\n{'='*60}\n  Starting stage: {stage_name.upper()}\n{'='*60}")
        try:
            result = stage_fns[stage_name]()
            stage_results[stage_name] = {"status": "success", **result,
                                          "wall_time_s": round(time.time()-t0, 1)}
            log.info(f"  Stage {stage_name} completed in {time.time()-t0:.1f}s")
        except Exception as e:
            log.error(f"  Stage {stage_name} FAILED: {e}", exc_info=True)
            stage_results[stage_name] = {"status": "failed", "error": str(e)}
            sys.exit(1)

    total_time = time.time() - pipeline_start
    log.info(f"\n{'='*60}")
    log.info(f"  Pipeline complete in {total_time:.1f}s")
    for s, r in stage_results.items():
        log.info(f"  {s:10s}: {r.get('status','?')!s:8s}  ({r.get('wall_time_s','?')}s)")

    summary_path = os.path.join(args.base_dir, "project", "data", "samples", "pipeline_summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as fp:
        json.dump({"stages": stage_results, "total_time_s": round(total_time, 1)}, fp, indent=2)
    log.info(f"  Summary → {summary_path}")


if __name__ == "__main__":
    main()
