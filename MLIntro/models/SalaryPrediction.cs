using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLIntro.models
{
     class SalaryPrediction
    {
        [ColumnName("Score")]
        public float PredictedSalary { get; set; }
    }
}
