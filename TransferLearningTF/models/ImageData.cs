using Microsoft.ML.Data;

namespace TransferLearningTF.models
{
    public class ImageData
    {
        [LoadColumn(0)]
        public string ImagePath;

        [LoadColumn(1)]
        public string Label;
    }
}
