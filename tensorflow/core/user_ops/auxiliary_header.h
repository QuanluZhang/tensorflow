#ifndef AUXILIARY_HEADER_H
#define AUXILIARY_HEADER_H

enum AuxDataType {
  // Not a legal value for DataType.  Used to indicate a DataType field
  // has not been set.
  DT_INVALID = 0,

  // Data types that all computation devices are expected to be
  // capable to support.
  DT_FLOAT = 1,
  DT_DOUBLE = 2,
  DT_INT32 = 3,
  DT_UINT8 = 4,
  DT_INT16 = 5,
  DT_INT8 = 6,
  DT_STRING = 7,
  DT_COMPLEX64 = 8,  // Single-precision complex
  DT_INT64 = 9,
  DT_BOOL = 10,
  DT_QINT8 = 11,     // Quantized int8
  DT_QUINT8 = 12,    // Quantized uint8
  DT_QINT32 = 13,    // Quantized int32
  DT_BFLOAT16 = 14,  // Float32 truncated to 16 bits.  Only for cast ops.
  DT_QINT16 = 15,    // Quantized int16
  DT_QUINT16 = 16,   // Quantized uint16
  DT_UINT16 = 17,
  DT_COMPLEX128 = 18,  // Double-precision complex
  DT_HALF = 19,
  DT_RESOURCE = 20,

  // TODO(josh11b): DT_GENERIC_PROTO = ??,
  // TODO(jeff,josh11b): DT_UINT64?  DT_UINT32?

  // Do not use!  These are only for parameters.  Every enum above
  // should have a corresponding value below (verified by types_test).
  DT_FLOAT_REF = 101,
  DT_DOUBLE_REF = 102,
  DT_INT32_REF = 103,
  DT_UINT8_REF = 104,
  DT_INT16_REF = 105,
  DT_INT8_REF = 106,
  DT_STRING_REF = 107,
  DT_COMPLEX64_REF = 108,
  DT_INT64_REF = 109,
  DT_BOOL_REF = 110,
  DT_QINT8_REF = 111,
  DT_QUINT8_REF = 112,
  DT_QINT32_REF = 113,
  DT_BFLOAT16_REF = 114,
  DT_QINT16_REF = 115,
  DT_QUINT16_REF = 116,
  DT_UINT16_REF = 117,
  DT_COMPLEX128_REF = 118,
  DT_HALF_REF = 119,
  DT_RESOURCE_REF = 120
};

#endif /*AUXILIARY_HEADER_H*/
