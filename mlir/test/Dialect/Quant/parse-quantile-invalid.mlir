// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// -----
// Illegal missing quantileType
// expected-error@+1 {{expected ':'}}
!qalias = !quant.quantile<u1:f32, {-1.0,1.0}:0.99872:127>
func.func @parse() -> !qalias {
  %0 = "foo"() : () -> !qalias
  return %0 : !qalias
}

// -----
// Illegal quantileType value
// expected-error@+1 {{illegal quantile type alias}}
!qalias = !quant.quantile<u2:f33:f32, {-1.0,1.0}:0.99872:127>
func.func @parse() -> !qalias {
  %0 = "foo"() : () -> !qalias
  return %0 : !qalias
}

// -----
// Illegal quantile array size
// expected-error@+1 {{quantiles array size needs to be equal to 2^(bit_size(storageType)), or (storageTypeMax - storageTypeMin + 1) when max and min differ from the type limits; expected: 256, found: 2}}
!qalias = !quant.quantile<i8:f16:f32, {-1.0,1.0}:0.99872:127>
func.func @parse() -> !qalias {
  %0 = "foo"() : () -> !qalias
  return %0 : !qalias
}

// -----
// Unrecognized token: trailing
// expected-error@+1 {{expected '>'}}
!qalias = !quant.quantile<i8<-4:3>:f16:f32, {-1.0,1.0}:0.99872:127 23>

// -----
// Unrecognized token: missing storage type maximum
// expected-error@+1 {{expected ':'}}
!qalias = !quant.quantile<i8<16>:f16:f32, {-1.0,1.0}:0.99872:127>

// -----
// Unrecognized token: missing closing angle bracket
// expected-error@+1 {{unbalanced '<' character in pretty dialect name}}
!qalias = !quant<quantile<i8<-4:3:f16:f32, {-1.0,1.0}:0.99872:127>>

// -----
// Unrecognized token: missing type colon
// expected-error@+1 {{expected ':'}}
!qalias = !quant.quantile<i8<-4:3>f16:f32, {-1.0,1.0}:0.99872:127>

// -----
// Unrecognized token: missing comma
// expected-error@+1 {{expected ','}}
!qalias = !quant.quantile<u2:f16:f32 {-1.0, -0.5, 0.5, 1.0}:0.99872:127>

// -----
// Unrecognized storage type: illegal prefix
// expected-error@+1 {{illegal quantized storage type alias}}
!qalias = !quant.quantile<int8<-4:3>:f16:f32, {-1.0,1.0}:0.99872:127>

// -----
// Unrecognized storage type: no width
// expected-error@+1 {{illegal quantized storage type alias}}
!qalias = !quant.quantile<i<-4:3>:f16:f32, {-1.0,1.0}:0.99872:127>

// -----
// Unrecognized storage type: storage size > 32
// expected-error@+1 {{illegal storage type size: 33}}
!qalias = !quant.quantile<i33:f16:f32, {-1.0,1.0}:0.99872:127>

// -----
// Unrecognized storage type: storage size < 0
// expected-error@+1 {{illegal quantized storage type alias}}
!qalias = !quant.quantile<i-1<-4:3>:f16:f32, {-1.0,1.0}:0.99872:127>

// -----
// Unrecognized storage type: storage size
// expected-error@+1 {{invalid integer width}}
!qalias = !quant.quantile<i123123123120<-4:3>:f16:f32, {-1.0,1.0}:0.99872:127>

// -----
// Illegal storage min/max: max - min < 0
// expected-error@+1 {{illegal storage min and storage max: (2:1)}}
!qalias = !quant.quantile<i8<2:1>:f16:f32, {-1.0,1.0}:0.99872:127>

// -----
// Illegal storage min/max: max - min == 0
// expected-error@+1 {{illegal storage min and storage max: (1:1)}}
!qalias = !quant.quantile<i8<1:1>:f16:f32, {-1.0,1.0}:0.99872:127>

// -----
// Illegal storage min/max: max > defaultMax
// expected-error@+1 {{illegal storage type maximum: 9}}
!qalias = !quant.quantile<i4<-1:9>:f16:f32, {-1.0,1.0}:0.99872:127>

// -----
// Illegal storage min/max: min < defaultMin
// expected-error@+1 {{illegal storage type minimum: -9}}
!qalias = !quant.quantile<i4<-9:1>:f16:f32, {-1.0,1.0}:0.99872:127>

// -----
// Illegal storage min/max: max > defaultMax
// expected-error@+1 {{illegal storage type maximum: 60000}}
!qalias = !quant.quantile<f8E5M2<-57344:60000>:f16:f32, {-1.0,1.0}:0.99872:127>

// -----
// Illegal storage min/max: min < defaultMin
// expected-error@+1 {{illegal storage type minimum: -60000}}
!qalias = !quant.quantile<f8E5M2<-60000:57344>:f16:f32, {-1.0,1.0}:0.99872:127>

// -----
// Illegal storage min/max: max > defaultMax
// expected-error@+1 {{illegal storage type maximum: 500}}
!qalias = !quant.quantile<f8E4M3FN<-448:500>:f16:f32, {-1.0,1.0}:0.99872:127>

// -----
// Illegal storage min/max: min < defaultMin
// expected-error@+1 {{illegal storage type minimum: -500}}
!qalias = !quant.quantile<f8E4M3FN<-500:448>:f16:f32, {-1.0,1.0}:0.99872:127>

// -----
// Illegal uniform params: invalid scale
// expected-error@+1 {{expected floating point literal}}
!qalias = !quant.quantile<i8<-4:3>:f16:f32, {-1.0,1.0}:abc:127>

// -----
// Illegal uniform params: invalid zero point separator
// expected-error@+1 {{expected '>'}}
!qalias = !quant.quantile<i8<-4:3>:f16:f32, {-1.0,1.0}:0.1abc>

// -----
// Illegal uniform params: missing zero point
// expected-error@+1 {{expected integer value}}
!qalias = !quant.quantile<i8<-4:3>:f16:f32, {-1.0,1.0}:0.1:>

// -----
// Illegal uniform params: invalid zero point
// expected-error@+1 {{expected integer value}}
!qalias = !quant.quantile<i8<-4:3>:f16:f32, {-1.0,1.0}:0.1:abc>

// -----
// Illegal expressed type: f33
// expected-error@+1 {{expected non-function type}}
!qalias = !quant.quantile<i8<-4:3>:f16:f33, {-1.0,1.0}:0.99872:127>

// -----
// Illegal scale: negative
// expected-error@+1 {{illegal scale: -1.000000}}
!qalias = !quant.quantile<i8<-4:3>:f16:f32, {-1.0,1.0}:-1.0:127>

// -----
// Illegal uniform params: missing quantized dimension
// expected-error@+1 {{expected integer value}}
!qalias = !quant.quantile<i8<-4:3>:f16:f32:, {-1.0,1.0}:{2.000000e+02:-19.987200e-01:1}>

// -----
// Illegal uniform params: unspecified quantized dimension, when multiple scales
// provided.
// expected-error@+1 {{expected floating point literal}}
!qalias = !quant.quantile<i8<-4:3>:f16:f32, {-1.0,1.0}:{2.000000e+02,-19.987200e-01:1}>

// -----
// Illegal quantile params: unspecified quantile values
// expected-error@+1 {{expected floating point literal}}
!qalias = !quant.quantile<i8<-4:3>:f16:f32, {}:0.99872:127>

// -----
// Illegal quantile params: missing quantile values
// expected-error@+1 {{expected floating point literal}}
!qalias = !quant.quantile<i8<-4:3>:f16:f32, {-1.0,}:0.99872:127>

// -----
// Illegal quantile params: missing colon separator
// expected-error@+1 {{expected ':'}}
!qalias = !quant.quantile<i8<-4:3>:f16:f32, {-1.0,1.0}0.99872:127>

// -----
// Illegal quantile params: unbalanced }
// expected-error@+1 {{unbalanced '{' character in pretty dialect name}}
!qalias = !quant.quantile<i8<-4:3>:f16:f32, {-1.0,1.0:0.99872:127>

// -----
// Illegal quantile params: missing {
// expected-error@+1 {{unbalanced '<' character in pretty dialect name}}
!qalias = !quant.quantile<i8<-4:3>:f16:f32, -1.0,1.0}:0.99872:127>
