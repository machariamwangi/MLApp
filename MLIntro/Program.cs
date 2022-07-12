// See https://aka.ms/new-console-template for more information
using Microsoft.ML.Data;
using Microsoft.ML;
namespace MLIntro
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello, World!");
            var context = new MLContext();
            var data = context.Data.LoadFromTextFile<HousingData>("./housing.csv", hasHeader: true, separatorChar: ',');

            var split = context.Data.TrainTestSplit(data, testFraction: 0.2);

            var features = split.TrainSet.Schema.Select(col => col.Name).Where(colName => colName != "Label" && colName != "OceanProximity").ToArray();

            var pipeline = context.Transforms.Text.FeaturizeText("Text", "OceanProximity")
                .Append(context.Transforms.Concatenate("Features", features))
                .Append(context.Transforms.Concatenate("Features", "Features", "Text"))
                .Append(context.Regression.Trainers.LbfgsPoissonRegression());

            var model = pipeline.Fit(split.TrainSet);

            var predictions = model.Transform(split.TestSet);

            var metrics = context.Regression.Evaluate(predictions);

            Console.WriteLine($"R^2 - {metrics.RSquared}");

        }
       
    }
}
