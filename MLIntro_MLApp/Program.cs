﻿
// This file was auto-generated by ML.NET Model Builder. 

using System;

namespace MLIntro_MLApp
{
    class Program
    {
        static void Main(string[] args)
        {
            // Create single instance of sample data from first line of dataset for model input
            MLIntro.ModelInput sampleData = new MLIntro.ModelInput()
            {
                Longitude = -122.23F,
                Latitude = 37.88F,
                Housing_median_age = 41F,
                Total_rooms = 880F,
                Total_bedrooms = 129F,
                Population = 322F,
                Households = 126F,
                Median_income = 8.3252F,
                Ocean_proximity = @"NEAR BAY",
            };

            // Make a single prediction on the sample data and print results
            var predictionResult = MLIntro.Predict(sampleData);

            Console.WriteLine("Using model to make single prediction -- Comparing actual Median_house_value with predicted Median_house_value from sample data...\n\n");


            Console.WriteLine($"Longitude: {-122.23F}");
            Console.WriteLine($"Latitude: {37.88F}");
            Console.WriteLine($"Housing_median_age: {41F}");
            Console.WriteLine($"Total_rooms: {880F}");
            Console.WriteLine($"Total_bedrooms: {129F}");
            Console.WriteLine($"Population: {322F}");
            Console.WriteLine($"Households: {126F}");
            Console.WriteLine($"Median_income: {8.3252F}");
            Console.WriteLine($"Median_house_value: {452600F}");
            Console.WriteLine($"Ocean_proximity: {@"NEAR BAY"}");


            Console.WriteLine($"\n\nPredicted Median_house_value: {predictionResult.Score}\n\n");
            Console.WriteLine("=============== End of process, hit any key to finish ===============");
            Console.ReadKey();
        }
    }
}