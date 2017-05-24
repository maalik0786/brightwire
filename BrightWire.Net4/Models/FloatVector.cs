﻿using ProtoBuf;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Xml;
using System.Linq;

namespace BrightWire.Models
{
    /// <summary>
    /// An protobuf serialised vector
    /// </summary>
    [ProtoContract]
    public class FloatVector
    {
        /// <summary>
        /// The data
        /// </summary>
        [ProtoMember(1)]
        public float[] Data { get; set; }

        /// <summary>
        /// The size of the vector
        /// </summary>
        public int Size { get { return Data?.Length ?? 0; } }

        /// <summary>
        /// ToString override
        /// </summary>
        public override string ToString()
        {
            var preview = String.Join("|", Data.Take(8));
            if (Size > 8)
                preview += "|...";
            return String.Format($"Vector ({Data.Length}):  {preview}");
        }

        /// <summary>
        /// Writes the data to an XML writer
        /// </summary>
        /// <param name="name">The name to give the data</param>
        /// <param name="writer">The writer to write to</param>
        public void WriteTo(string name, XmlWriter writer)
        {
            writer.WriteStartElement(name ?? "vector");

            if (Data != null)
                writer.WriteValue(String.Join("|", Data));
            writer.WriteEndElement();
        }

        /// <summary>
        /// Writes the data to a binary writer
        /// </summary>
        /// <param name="writer"></param>
        public void WriteTo(BinaryWriter writer)
        {
            writer.Write(Size);
            if (Data != null) {
                foreach (var val in Data)
                    writer.Write(val);
            }
        }

        /// <summary>
        /// Creates a float array from a binary reader
        /// </summary>
        /// <param name="reader">The binary reader</param>
        public static FloatVector ReadFrom(BinaryReader reader)
        {
            var len = reader.ReadInt32();
            var ret = new float[len];
            for (var i = 0; i < len; i++)
                ret[i] = reader.ReadSingle();
            return new FloatVector {
                Data = ret
            };
        }

        /// <summary>
        /// Converts the vector to XML
        /// </summary>
        public string Xml
        {
            get
            {
                var sb = new StringBuilder();
                var settings = new XmlWriterSettings {
                    OmitXmlDeclaration = true
                };
                using (var writer = XmlWriter.Create(sb, settings))
                    WriteTo(null, writer);
                return sb.ToString();
            }
        }

        public bool IsEqualTo(FloatVector vector, IEqualityComparer<float> comparer = null)
        {
            if (vector == null || Size != vector.Size)
                return false;

            var comparer2 = comparer ?? EqualityComparer<float>.Default;
            for (int i = 0, len = Size; i < len; i++) {
                if (!comparer2.Equals(Data[i], vector.Data[i]))
                    return false;
            }
            return true;
        }

        public int MaximumIndex()
        {
            var ret = 0;
            var max = float.MinValue;
            for(int i = 0, len = Size; i < len; i++) {
                var val = Data[i];
                if(val > max) {
                    max = Data[i];
                    ret = i;
                }
            }
            return ret;
        }
    }
}
