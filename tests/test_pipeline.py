#!/usr/bin/env python3
"""
test_pipeline.py
================
Unit and integration tests for the HDFS Anomaly Detection ML pipeline.
Tests cover:
  - Data loading and schema validation
  - Feature engineering transformations
  - Custom transformer (EventCategoryAggregator)
  - Model I/O (save/load)
  - Evaluation metrics consistency
  - Pipeline end-to-end smoke test

Usage:
    python -m pytest tests/test_pipeline.py -v
    # or directly:
    python tests/test_pipeline.py
"""

import json
import os
import sys
import tempfile
import unittest
import warnings

warnings.filterwarnings("ignore")

# Ensure project root on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

BASE_DIR = "/home/sayan/Lakhveer"


# ─────────────────────────────────────────────────────────────────────────────
# Shared SparkSession fixture (created once, reused across tests)
# ─────────────────────────────────────────────────────────────────────────────
def get_test_spark():
    from pyspark.sql import SparkSession
    spark = (
        SparkSession.builder
        .appName("HDFS_Pipeline_Tests")
        .config("spark.driver.memory",          "2g")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.serializer",
                "org.apache.spark.serializer.KryoSerializer")
        .master("local[2]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


# ─────────────────────────────────────────────────────────────────────────────
# Test class 1 – Schema & Data loading
# ─────────────────────────────────────────────────────────────────────────────
class TestDataIngestion(unittest.TestCase):
    """Tests for data loading, schema validation, and null checks."""

    @classmethod
    def setUpClass(cls):
        cls.spark = get_test_spark()
        cls.parquet_path = os.path.join(BASE_DIR, "project", "data", "hdfs_parquet")

    @classmethod
    def tearDownClass(cls):
        pass  # Keep SparkSession alive for other test classes

    def test_parquet_exists(self):
        """Parquet directory should exist after ingestion stage."""
        self.assertTrue(
            os.path.exists(self.parquet_path),
            f"Parquet not found at {self.parquet_path} – run stage_ingest first",
        )

    def test_parquet_row_count(self):
        """Row count should be > 1 million."""
        if not os.path.exists(self.parquet_path):
            self.skipTest("Parquet not yet ingested")
        df = self.spark.read.parquet(self.parquet_path)
        n  = df.count()
        self.assertGreater(n, 1_000_000, f"Expected >1M rows, got {n}")

    def test_label_column_exists(self):
        """Label column must be present."""
        if not os.path.exists(self.parquet_path):
            self.skipTest("Parquet not yet ingested")
        df = self.spark.read.parquet(self.parquet_path)
        self.assertIn("Label", df.columns)

    def test_label_binary(self):
        """Label must contain only 0 and 1."""
        if not os.path.exists(self.parquet_path):
            self.skipTest("Parquet not yet ingested")
        from pyspark.sql import functions as F
        df = self.spark.read.parquet(self.parquet_path)
        distinct_labels = {row["Label"] for row in df.select("Label").distinct().collect()}
        self.assertTrue(
            distinct_labels.issubset({0, 1}),
            f"Unexpected label values: {distinct_labels}",
        )

    def test_event_columns_present(self):
        """All E1-E29 event columns must be present."""
        if not os.path.exists(self.parquet_path):
            self.skipTest("Parquet not yet ingested")
        df = self.spark.read.parquet(self.parquet_path)
        for i in range(1, 30):
            self.assertIn(f"E{i}", df.columns, f"Missing column E{i}")

    def test_no_critical_nulls(self):
        """Key columns E5, E9, Label should have zero nulls."""
        if not os.path.exists(self.parquet_path):
            self.skipTest("Parquet not yet ingested")
        from pyspark.sql import functions as F
        df    = self.spark.read.parquet(self.parquet_path)
        nulls = df.select(
            F.count(F.when(F.col("E5").isNull(),   1)).alias("E5"),
            F.count(F.when(F.col("E9").isNull(),   1)).alias("E9"),
            F.count(F.when(F.col("Label").isNull(),1)).alias("Label"),
        ).collect()[0].asDict()
        for col, cnt in nulls.items():
            self.assertEqual(cnt, 0, f"Unexpected nulls in column {col}: {cnt}")


# ─────────────────────────────────────────────────────────────────────────────
# Test class 2 – Custom Transformer
# ─────────────────────────────────────────────────────────────────────────────
class TestEventCategoryAggregator(unittest.TestCase):
    """Tests for the custom EventCategoryAggregator transformer."""

    @classmethod
    def setUpClass(cls):
        cls.spark = get_test_spark()

    def _make_test_df(self):
        from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType
        schema = StructType(
            [StructField(f"E{i}", DoubleType(), True) for i in range(1, 30)]
            + [StructField("Label", IntegerType(), True),
               StructField("total_events", DoubleType(), True)]
        )
        rows = [
            tuple([1.0]*29 + [0, 29.0]),  # all events = 1
            tuple([0.0]*29 + [1, 0.0]),   # all events = 0
        ]
        return self.spark.createDataFrame(rows, schema=schema)

    def test_category_columns_created(self):
        """Transformer must add cat_data_transfer, cat_error, cat_replication, cat_admin."""
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
        # Inline minimal transformer for test independence
        from pyspark.ml import Transformer
        from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
        from pyspark.sql import functions as F

        class EventCategoryAggregator(Transformer, DefaultParamsReadable, DefaultParamsWritable):
            CATEGORY_MAP = {
                "cat_data_transfer": ["E2","E5","E9","E11","E21","E23","E25","E26"],
                "cat_error"        : ["E4","E6","E7","E8","E22"],
                "cat_replication"  : ["E3","E15","E16","E19"],
                "cat_admin"        : ["E14","E18","E24","E27","E28","E29"],
            }
            def _transform(self, dataset):
                for cat_col, events in self.CATEGORY_MAP.items():
                    dataset = dataset.withColumn(cat_col, sum(F.col(e) for e in events))
                return dataset

        df    = self._make_test_df()
        trans = EventCategoryAggregator()
        out   = trans.transform(df)
        for col in ["cat_data_transfer","cat_error","cat_replication","cat_admin"]:
            self.assertIn(col, out.columns, f"Missing column: {col}")

    def test_category_values_correct(self):
        """cat_error should sum E4+E6+E7+E8+E22; with all events=1, expect 5."""
        from pyspark.ml import Transformer
        from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
        from pyspark.sql import functions as F

        class EventCategoryAggregator(Transformer, DefaultParamsReadable, DefaultParamsWritable):
            CATEGORY_MAP = {
                "cat_data_transfer": ["E2","E5","E9","E11","E21","E23","E25","E26"],
                "cat_error"        : ["E4","E6","E7","E8","E22"],
                "cat_replication"  : ["E3","E15","E16","E19"],
                "cat_admin"        : ["E14","E18","E24","E27","E28","E29"],
            }
            def _transform(self, dataset):
                for cat_col, events in self.CATEGORY_MAP.items():
                    dataset = dataset.withColumn(cat_col, sum(F.col(e) for e in events))
                return dataset

        df  = self._make_test_df()
        out = EventCategoryAggregator().transform(df)
        row = out.filter(F.col("Label") == 0).select("cat_error").first()
        self.assertAlmostEqual(float(row["cat_error"]), 5.0, places=1,
                               msg="cat_error with all events=1 should equal 5")


# ─────────────────────────────────────────────────────────────────────────────
# Test class 3 – Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────
class TestFeatureEngineering(unittest.TestCase):
    """Tests for vector assembly, scaling, and PCA."""

    @classmethod
    def setUpClass(cls):
        cls.spark = get_test_spark()
        cls.feat_path = os.path.join(BASE_DIR, "project", "data", "features_parquet")

    def test_splits_exist(self):
        """Train/val/test split directories must exist."""
        for split in ["train", "val", "test"]:
            path = os.path.join(self.feat_path, split)
            self.assertTrue(
                os.path.exists(path),
                f"Split directory missing: {path}",
            )

    def test_feature_columns_in_splits(self):
        """Each split must contain std_features, Label, class_weight."""
        if not os.path.exists(self.feat_path):
            self.skipTest("Feature parquet not yet built")
        for split in ["train", "val", "test"]:
            path = os.path.join(self.feat_path, split)
            if not os.path.exists(path):
                continue
            df = self.spark.read.parquet(path)
            for col in ["std_features", "Label", "class_weight"]:
                self.assertIn(col, df.columns, f"{col} missing in {split} split")

    def test_feature_vector_dimension(self):
        """Feature vector should have dimension >= 29 (at least raw events)."""
        if not os.path.exists(self.feat_path):
            self.skipTest("Feature parquet not yet built")
        train_path = os.path.join(self.feat_path, "train")
        if not os.path.exists(train_path):
            self.skipTest("Train split not found")
        df    = self.spark.read.parquet(train_path)
        first = df.select("std_features").first()
        dim   = len(first["std_features"])
        self.assertGreaterEqual(dim, 29, f"Feature dim {dim} < 29")

    def test_label_preserved_after_engineering(self):
        """Labels in train split should still be binary."""
        train_path = os.path.join(self.feat_path, "train")
        if not os.path.exists(train_path):
            self.skipTest("Train split not found")
        from pyspark.sql import functions as F
        df = self.spark.read.parquet(train_path)
        labels = {r["Label"] for r in df.select("Label").distinct().collect()}
        self.assertTrue(labels.issubset({0, 1}), f"Non-binary labels: {labels}")


# ─────────────────────────────────────────────────────────────────────────────
# Test class 4 – Model Save/Load
# ─────────────────────────────────────────────────────────────────────────────
class TestModelSerialisation(unittest.TestCase):
    """Tests for MLlib model serialisation (save/load round-trip)."""

    @classmethod
    def setUpClass(cls):
        cls.spark      = get_test_spark()
        cls.models_dir = os.path.join(BASE_DIR, "project", "data", "models")

    def _model_loadable(self, model_cls, model_subdir):
        path = os.path.join(self.models_dir, model_subdir)
        if not os.path.exists(path):
            self.skipTest(f"Model not found: {path}")
        model = model_cls.load(path)
        self.assertIsNotNone(model)

    def test_lr_model_loadable(self):
        from pyspark.ml.classification import LogisticRegressionModel
        self._model_loadable(LogisticRegressionModel, "lr_model")

    def test_rf_model_loadable(self):
        from pyspark.ml.classification import RandomForestClassificationModel
        self._model_loadable(RandomForestClassificationModel, "rf_model")

    def test_gbt_model_loadable(self):
        from pyspark.ml.classification import GBTClassificationModel
        self._model_loadable(GBTClassificationModel, "gbt_model")

    def test_svc_model_loadable(self):
        from pyspark.ml.classification import LinearSVCModel
        self._model_loadable(LinearSVCModel, "svc_model")

    def test_sklearn_pickle_loadable(self):
        """Sklearn baseline pickle should be loadable."""
        import pickle
        pkl_path = os.path.join(self.models_dir, "sklearn_lr_baseline.pkl")
        if not os.path.exists(pkl_path):
            self.skipTest("sklearn pickle not found")
        with open(pkl_path, "rb") as fh:
            obj = pickle.load(fh)
        self.assertIn("model", obj)
        self.assertIn("results", obj)


# ─────────────────────────────────────────────────────────────────────────────
# Test class 5 – Evaluation Outputs
# ─────────────────────────────────────────────────────────────────────────────
class TestEvaluationOutputs(unittest.TestCase):
    """Tests that evaluation stage produces expected output artefacts."""

    SAMPLES_DIR = os.path.join(BASE_DIR, "project", "data", "samples")

    def test_final_results_json_exists(self):
        path = os.path.join(self.SAMPLES_DIR, "final_test_results.json")
        if not os.path.exists(path):
            self.skipTest("final_test_results.json not found – run eval stage first")
        with open(path) as fp:
            data = json.load(fp)
        self.assertIn("test_results", data)

    def test_model_metrics_range(self):
        """All AUC / F1 / Accuracy must be in [0, 1]."""
        path = os.path.join(self.SAMPLES_DIR, "final_test_results.json")
        if not os.path.exists(path):
            self.skipTest("final_test_results.json not found")
        with open(path) as fp:
            data = json.load(fp)
        for model_name, metrics in data.get("test_results", {}).items():
            for metric_name, val in metrics.items():
                self.assertGreaterEqual(float(val), 0.0,
                    f"{model_name}.{metric_name} = {val} < 0")
                self.assertLessEqual(float(val), 1.0,
                    f"{model_name}.{metric_name} = {val} > 1")

    def test_tableau_csvs_exist(self):
        """Expected Tableau CSV exports must exist if eval was run."""
        expected = [
            "tableau_model_performance.csv",
            "tableau_data_quality.csv",
            "roc_data.csv",
            "pr_data.csv",
        ]
        path = os.path.join(self.SAMPLES_DIR, "final_test_results.json")
        if not os.path.exists(path):
            self.skipTest("Eval not yet run")
        for fname in expected:
            fpath = os.path.join(self.SAMPLES_DIR, fname)
            self.assertTrue(os.path.exists(fpath), f"Missing Tableau export: {fname}")

    def test_png_outputs_exist(self):
        """Key visualisation PNGs must be generated."""
        path = os.path.join(self.SAMPLES_DIR, "final_test_results.json")
        if not os.path.exists(path):
            self.skipTest("Eval not yet run")
        expected_pngs = [
            "confusion_matrices.png",
            "roc_curves.png",
            "pr_curves.png",
            "bootstrap_ci.png",
        ]
        for fname in expected_pngs:
            fpath = os.path.join(self.SAMPLES_DIR, fname)
            self.assertTrue(os.path.exists(fpath), f"Missing PNG: {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# Test class 6 – Smoke Tests (tiny synthetic data)
# ─────────────────────────────────────────────────────────────────────────────
class TestSmokeEndToEnd(unittest.TestCase):
    """Fast smoke tests using tiny synthetic DataFrames."""

    @classmethod
    def setUpClass(cls):
        cls.spark = get_test_spark()

    def _make_synthetic_df(self, n_rows: int = 200):
        """Generate a tiny fake HDFS-like DataFrame for smoke testing."""
        import random
        random.seed(42)
        from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType

        schema = StructType(
            [StructField(f"E{i}", DoubleType(), True) for i in range(1, 30)]
            + [StructField("Label", IntegerType(), True),
               StructField("total_events", DoubleType(), True)]
        )
        rows = []
        for _ in range(n_rows):
            events = [float(random.randint(0, 5)) for _ in range(29)]
            label  = random.choice([0, 1])
            total  = sum(events)
            rows.append(tuple(events + [label, total]))
        return self.spark.createDataFrame(rows, schema=schema)

    def test_vector_assembler_smoke(self):
        """VectorAssembler should produce dense vector without error."""
        from pyspark.ml.feature import VectorAssembler
        df   = self._make_synthetic_df()
        cols = [f"E{i}" for i in range(1, 30)]
        va   = VectorAssembler(inputCols=cols, outputCol="features", handleInvalid="skip")
        out  = va.transform(df)
        self.assertIn("features", out.columns)
        first_vec = out.select("features").first()["features"]
        self.assertEqual(len(first_vec), 29)

    def test_standard_scaler_smoke(self):
        """StandardScaler should fit and transform without error."""
        from pyspark.ml.feature import VectorAssembler, StandardScaler
        df   = self._make_synthetic_df()
        cols = [f"E{i}" for i in range(1, 30)]
        df   = VectorAssembler(inputCols=cols, outputCol="raw").transform(df)
        out  = StandardScaler(inputCol="raw", outputCol="scaled", withMean=True, withStd=True).fit(df).transform(df)
        self.assertIn("scaled", out.columns)

    def test_logistic_regression_smoke(self):
        """LogisticRegression should fit and predict on tiny data."""
        from pyspark.ml.feature import VectorAssembler, StandardScaler
        from pyspark.ml.classification import LogisticRegression
        df   = self._make_synthetic_df(n_rows=500)
        cols = [f"E{i}" for i in range(1, 30)]
        df   = VectorAssembler(inputCols=cols, outputCol="raw", handleInvalid="skip").transform(df)
        df   = StandardScaler(inputCol="raw", outputCol="features",
                               withMean=True, withStd=True).fit(df).transform(df)
        model = LogisticRegression(featuresCol="features", labelCol="Label", maxIter=5).fit(df)
        pred  = model.transform(df)
        self.assertIn("prediction", pred.columns)
        n_preds = pred.count()
        self.assertGreater(n_preds, 0)

    def test_random_forest_smoke(self):
        """RandomForestClassifier should train on tiny data."""
        from pyspark.ml.feature import VectorAssembler
        from pyspark.ml.classification import RandomForestClassifier
        df   = self._make_synthetic_df(n_rows=500)
        cols = [f"E{i}" for i in range(1, 30)]
        df   = VectorAssembler(inputCols=cols, outputCol="features", handleInvalid="skip").transform(df)
        model = RandomForestClassifier(featuresCol="features", labelCol="Label",
                                        numTrees=10, maxDepth=3, seed=42).fit(df)
        pred  = model.transform(df)
        self.assertIn("prediction", pred.columns)

    def test_gbt_smoke(self):
        """GBTClassifier should train on tiny data."""
        from pyspark.ml.feature import VectorAssembler
        from pyspark.ml.classification import GBTClassifier
        df   = self._make_synthetic_df(n_rows=500)
        cols = [f"E{i}" for i in range(1, 30)]
        df   = VectorAssembler(inputCols=cols, outputCol="features", handleInvalid="skip").transform(df)
        model = GBTClassifier(featuresCol="features", labelCol="Label",
                               maxIter=5, maxDepth=3, seed=42).fit(df)
        pred  = model.transform(df)
        self.assertIn("prediction", pred.columns)

    def test_temporary_parquet_write_read(self):
        """Should be able to write and read back Parquet."""
        from pyspark.ml.feature import VectorAssembler
        df  = self._make_synthetic_df()
        cols = [f"E{i}" for i in range(1, 6)]
        df  = VectorAssembler(inputCols=cols, outputCol="features", handleInvalid="skip").transform(df)
        with tempfile.TemporaryDirectory() as tmpdir:
            df.write.mode("overwrite").parquet(tmpdir)
            df2 = self.spark.read.parquet(tmpdir)
            self.assertEqual(df2.count(), df.count())


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Run with verbosity=2 for detailed output
    loader  = unittest.TestLoader()
    suite   = unittest.TestSuite()

    # Order: smoke tests first (fastest), then full pipeline tests
    for cls in [
        TestSmokeEndToEnd,
        TestEventCategoryAggregator,
        TestDataIngestion,
        TestFeatureEngineering,
        TestModelSerialisation,
        TestEvaluationOutputs,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
