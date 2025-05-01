package com.cs643.assignment2;

import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.feature.StandardScaler;
import org.apache.spark.ml.classification.GBTClassificationModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.classification.RandomForestClassificationModel;

import java.util.Arrays;

public class Test {
    // Constants for default paths
    private static final String DEFAULT_MODEL_PATH = "s3:/cs643-wine-date-pn338/best_model/";
    private static final String DEFAULT_TEST_PATH = "s3:/cs643-wine-date-pn338/ValidationDataset.csv";
    
    public static void main(String[] args) {
        // Initialize paths from command line args or use defaults
        String modelPath = args.length > 0 ? args[0] : DEFAULT_MODEL_PATH;
        String testFilePath = args.length > 1 ? args[1] : DEFAULT_TEST_PATH;

        // Create Spark session with descriptive name
        SparkSession spark = SparkSession.builder()
                .appName("Wine Quality Prediction Test")
                .getOrCreate();

        try {
            // Load and preprocess the validation dataset
            Dataset<Row> validationData = loadAndProcessData(spark, testFilePath);
            
            // Debug: Display scaled features
            System.out.println("Scaled Features Preview:");
            validationData.select("scaledFeatures").show(false);

            // Load the pre-trained model and generate predictions
            RandomForestClassificationModel loadedModel = RandomForestClassificationModel.load(modelPath);
            Dataset<Row> predictions = loadedModel.transform(validationData);

            // Display prediction results
            System.out.println("Prediction Results:");
            predictions.show();

            // Calculate and display model performance metrics
            double f1Score = evaluateModel(predictions);
            System.out.println("Model Performance - F1 Score: " + f1Score);
        } finally {
            spark.stop();
        }
    }

    private static Dataset<Row> loadAndProcessData(SparkSession spark, String filePath) {
        // Feature columns for the wine quality dataset
        String[] featureColumns = {
            "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
            "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
            "pH", "sulphates", "alcohol", "quality"
        };

        // Read CSV file with wine quality data
        Dataset<Row> df = spark.read()
                .option("header", "true")
                .option("sep", ";")
                .option("inferSchema", "true")
                .csv(filePath)
                .toDF(featureColumns);

        // Create and apply feature processing pipeline
        return createAndApplyPipeline(df, featureColumns);
    }

    private static Dataset<Row> createAndApplyPipeline(Dataset<Row> df, String[] columns) {
        // Configure feature assembly and scaling
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(Arrays.copyOfRange(columns, 0, columns.length - 1))
                .setOutputCol("features");
                
        StandardScaler scaler = new StandardScaler()
                .setInputCol("features")
                .setOutputCol("scaledFeatures")
                .setWithStd(true)
                .setWithMean(true);

        // Create and execute pipeline
        Pipeline pipeline = new Pipeline().setStages(new PipelineStage[] {assembler, scaler});
        return pipeline.fit(df).transform(df);
    }

    private static double evaluateModel(Dataset<Row> predictions) {
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("quality")
                .setPredictionCol("prediction")
                .setMetricName("f1");
        return evaluator.evaluate(predictions);
    }
}
