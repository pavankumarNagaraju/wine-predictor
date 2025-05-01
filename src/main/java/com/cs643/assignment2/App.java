package com.cs643.assignment2;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.StandardScaler;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.util.Arrays;

public class App {
    private static final String MODEL_SAVE_PATH = "s3a://cs643-wine-date-pn338/best_model";

    public static void main(String[] args) {
        if (args.length < 2) {
            System.err.println("Usage: App <training_file> <validation_file>");
            System.exit(1);
        }

        String trainPath = args[0];
        String validationPath = args[1];

        SparkSession spark = SparkSession.builder()
                .appName("Wine Quality Prediction")
                .getOrCreate();

        Dataset<Row> trainDF = loadAndProcessData(spark, trainPath);
        Dataset<Row> valDF = loadAndProcessData(spark, validationPath);

        LogisticRegression lr = new LogisticRegression()
                .setLabelCol("quality")
                .setFeaturesCol("scaledFeatures");

        LogisticRegressionModel lrModel = lr.fit(trainDF);

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("quality")
                .setPredictionCol("prediction");

        double lrF1Train = evaluator.evaluate(lrModel.transform(trainDF));
        double lrF1Val = evaluator.evaluate(lrModel.transform(valDF));

        System.out.println("Logistic Regression F1 (Train): " + lrF1Train);
        System.out.println("Logistic Regression F1 (Validation): " + lrF1Val);

        RandomForestClassifier rf = new RandomForestClassifier()
                .setLabelCol("quality")
                .setFeaturesCol("scaledFeatures");

        ParamMap[] paramGrid = new ParamGridBuilder()
                .addGrid(rf.numTrees(), new int[]{10, 20})
                .addGrid(rf.maxDepth(), new int[]{5, 10})
                .build();

        CrossValidator cv = new CrossValidator()
                .setEstimator(rf)
                .setEvaluator(evaluator)
                .setEstimatorParamMaps(paramGrid)
                .setNumFolds(3);

        CrossValidatorModel cvModel = cv.fit(trainDF);
        double bestF1 = cvModel.avgMetrics()[0];

        System.out.println("Best Model F1 Score: " + bestF1);

        try {
            RandomForestClassificationModel bestModel = (RandomForestClassificationModel) cvModel.bestModel();
            bestModel.write().overwrite().save(MODEL_SAVE_PATH);
            System.out.println("Best model saved to: " + MODEL_SAVE_PATH);
        } catch (Exception e) {
            System.err.println("Failed to save best model: " + e.getMessage());
        }

        spark.stop();
    }

    private static Dataset<Row> loadAndProcessData(SparkSession spark, String path) {
        String[] columns = {
                "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
                "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
                "pH", "sulphates", "alcohol", "quality"
        };

        Dataset<Row> df = spark.read()
                .option("header", "true")
                .option("sep", ";")
                .option("inferSchema", "true")
                .csv(path)
                .toDF(columns);

        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(Arrays.copyOfRange(columns, 0, columns.length - 1))
                .setOutputCol("features");

        StandardScaler scaler = new StandardScaler()
                .setInputCol("features")
                .setOutputCol("scaledFeatures")
                .setWithStd(true)
                .setWithMean(true);

        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[]{assembler, scaler});

        return pipeline.fit(df).transform(df);
    }
}
