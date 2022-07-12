using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using Microsoft.ML;
using static Microsoft.ML.DataOperationsCatalog;
using Microsoft.ML.Vision;



namespace TransferLearningTF
{

    class Program
    {

        static void Main(string[] args)
        {
            var projectDirectory = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../"));
            var workspaceRelativePath = Path.Combine(projectDirectory, "workspace");
            var assetsRelativePath = Path.Combine(projectDirectory, "assets");
            MLContext mlContext = new MLContext();

            //loading images from dir
            IEnumerable<ImageData> LoadImagesFromDirectory(string folder, bool useFolderNameAsLabel = true)
            {
                var files = Directory.GetFiles(folder, "*", searchOption: SearchOption.AllDirectories);
                foreach (var file in files)
                {
                    if ((Path.GetExtension(file) != ".jpg") && (Path.GetExtension(file) != ".png"))
                        continue;
                    var label = Path.GetFileName(file);

                    if (useFolderNameAsLabel)
                        label = Directory.GetParent(file).Name;
                    else
                    {
                        for (int index = 0; index < label.Length; index++)
                        {
                            if (!char.IsLetter(label[index]))
                            {
                                label = label.Substring(0, index);
                                break;
                            }
                        }
                    }

                    yield return new ImageData()
                    {
                        ImagePath = file,
                        Label = label
                    };


                }
            }

            // get the list of images used for training after initialzing the mlcontext variable
            IEnumerable<ImageData> images = LoadImagesFromDirectory(folder: assetsRelativePath, useFolderNameAsLabel: true);

            IDataView imageData = mlContext.Data.LoadFromEnumerable(images);

            IDataView shuffledData = mlContext.Data.ShuffleRows(imageData);


            //making the model pipeline to transform data that will be used to train the machine  or model in this case.

            var preprocessingPipeline = mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Label", outputColumnName: "LabelAsKey")
                                        .Append(mlContext.Transforms.LoadRawImageBytes(outputColumnName: "Image", imageFolder: assetsRelativePath, inputColumnName: "ImagePath"));

            IDataView preProcessedData = preprocessingPipeline.Fit(shuffledData).Transform(shuffledData);

            TrainTestData trainSplit = mlContext.Data.TrainTestSplit(data: preProcessedData, testFraction: 0.3);
            TrainTestData validationTestSplit = mlContext.Data.TrainTestSplit(trainSplit.TestSet);

            IDataView trainSet = trainSplit.TrainSet;
            IDataView validationSet = validationTestSplit.TrainSet;
            IDataView testSet = validationTestSplit.TestSet;


            var classifierOptions = new ImageClassificationTrainer.Options()
            {
                FeatureColumnName = "Image",
                LabelColumnName = "LabelAsKey",
                ValidationSet = validationSet,
                Arch = ImageClassificationTrainer.Architecture.ResnetV2101,
                MetricsCallback = (metrics) => Console.WriteLine(metrics),
                TestOnTrainSet = false,
                ReuseTrainSetBottleneckCachedValues = true,
                ReuseValidationSetBottleneckCachedValues = true
            };

            var trainingPipeline = mlContext.MulticlassClassification.Trainers.ImageClassification(classifierOptions)
                                   .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            //We use now the fit method to train our created  model

            ITransformer trainedModel = trainingPipeline.Fit(trainSet);


           // display prediction information in the console.
             static void OutputPrediction(ModelOutPut prediction)
            {
                string imageName = Path.GetFileName(prediction.ImagePath);
                Console.WriteLine($"Image: {imageName} | Actual Value: {prediction.Label} | Predicted Value: {prediction.PredictedLabel}");
            }

            void ClassifySingleImage(MLContext mlContext, IDataView data, ITransformer trainedModel)
            {
                PredictionEngine<ModelInput, ModelOutPut> predictionEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutPut>(trainedModel);
                ModelInput image = mlContext.Data.CreateEnumerable<ModelInput>(data, reuseRowObject: true).First();
                ModelOutPut prediction = predictionEngine.Predict(image);
                Console.WriteLine("Classifying single image");
                OutputPrediction(prediction);
            }

            ClassifySingleImage(mlContext, testSet, trainedModel);


            void ClassifyImages(MLContext mlContext, IDataView data, ITransformer trainedModel)
            {
                IDataView predictionData = trainedModel.Transform(data);
                IEnumerable<ModelOutPut> predictions = mlContext.Data.CreateEnumerable<ModelOutPut>(predictionData, reuseRowObject: true).Take(10);
                Console.WriteLine("Classifying multiple images");
                foreach (var prediction in predictions)
                {
                    OutputPrediction(prediction);
                }
            }

            ClassifyImages(mlContext, testSet, trainedModel);
        }

    }

    class ImageData
    {
        public string ImagePath { get; set; }
        public string Label { get; set; }
    }

    class ModelInput
    {
        public byte[] Image { get; set; }
        public UInt32 LabelAsKey { get; set; }
        public string ImagePath { get; set; }
        public string Label { get; set; }
    }

    class ModelOutPut
    {
        public string ImagePath { get; set; }

        public string Label { get; set; }

        public string PredictedLabel { get; set; }
    }



}
