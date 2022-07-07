using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations.Schema;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLIntro.models
{
     class SalaryData
    {
        [Column("0")]
        public float YearsExperience { get; set; }
        [LoadColumn(1), ColumnName("label")]
        public float Salary { get; set; }
       
    }
}
