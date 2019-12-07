﻿using System;
using System.Collections.Generic;
using System.Text;

namespace BrightData
{
    public class Consts
    {
        public const int DataTableVersion = 1;
        public const int MaxDistinct = 131072 * 4;
        public const int DefaultMemoryCacheSize = 1024 * 1048576;

        //public const string Id = "Id";
        public const string Index = "Index";
        public const string Name = "Name";
        public const string Type = "Type";
        public const string IsNumeric = "IsNumeric";
        public const string IsTarget = "IsTarget";

        public const string HasUnique = "HasUnique";

        public const string HasBeenAnalysed = "HasBeenAnalysed";
        public const string Mode = "Mode";
        public const string NumDistinct = "NumDistinct";
        public const string MinDate = "MinDate";
        public const string MaxDate = "MaxDate";
        public const string MinIndex = "MinIndex";
        public const string MaxIndex = "MaxIndex";
        public const string MinLength = "MinLength";
        public const string MaxLength = "MaxLength";
        public const string XDimension = "XDimension";
        public const string YDimension = "YDimension";
        public const string ZDimension = "ZDimension";
        public const string L1Norm = "L1Norm";
        public const string L2Norm = "L2Norm";
        public const string Min = "Min";
        public const string Max = "Max";
        public const string Mean = "Mean";
        public const string Variance = "Variance";
        public const string StdDev = "StdDev";
        public const string Median = "Median";
        public const string FrequencyPrefix = "Frequency:";
        public const string FrequencyRangePrefix = "FrequencyRange:";

        public static readonly string[] StandardMetaData = new[] { Index, Name, Type, IsNumeric, IsTarget };
    }
}