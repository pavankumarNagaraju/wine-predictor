FROM openjdk:17

WORKDIR /app

COPY target/wine-predictor-1.0.jar wine-predictor-1.0.jar

ENTRYPOINT ["java", "-cp", "wine-predictor-1.0.jar", "com.cs643.assignment2.App"]
