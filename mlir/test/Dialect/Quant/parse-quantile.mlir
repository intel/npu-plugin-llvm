// RUN: mlir-opt -allow-unregistered-dialect %s -split-input-file | FileCheck %s

// -----
// All per-layer params specified:
//   [signed] storageType, storageTypeMin, storageTypeMax, expressedType, scale, zeroPoint
// CHECK: !quant.quantile<i8<-8:7>:f32, {-1.000000e+00,1.000000e+00}:9.987200e-01:127>
!qalias = !quant.quantile<i8<-8:7>:f32, {-1.0,1.0}:0.99872:127>
func.func @parse() -> !qalias {
  %0 = "foo"() : () -> !qalias
  return %0 : !qalias
}

// -----
// Trailing whitespace.
// CHECK: !quant.quantile<i8<-8:7>:f32, {-1.000000e+00,1.000000e+00}:9.987200e-01:127>
!qalias = !quant.quantile<i8<-8:7>:f32, {-1.0,1.0}:0.99872:127  >
func.func @parse() -> !qalias {
  %0 = "foo"() : () -> !qalias
  return %0 : !qalias
}

// -----
// Default min/max value optimization for integers.
// CHECK: !quant.quantile<i8:f32, {-1.000000e+00,1.000000e+00}:9.987200e-01:127>
!qalias = !quant.quantile<i8<-128:127>:f32, {-1.0,1.0}:0.99872:127  >
func.func @parse() -> !qalias {
  %0 = "foo"() : () -> !qalias
  return %0 : !qalias
}

// -----
// Default min/max value optimization for f8E5M2.
// CHECK: !quant.quantile<f8E5M2:f32, {-1.000000e+00,1.000000e+00}:9.987200e-01:127>
!qalias = !quant.quantile<f8E5M2<-57344:57344>:f32, {-1.0,1.0}:0.99872:127  >
func.func @parse() -> !qalias {
  %0 = "foo"() : () -> !qalias
  return %0 : !qalias
}

// -----
// Default min/max value optimization for f8E4M3FN.
// CHECK: !quant.quantile<f8E4M3FN:f32, {-1.000000e+00,1.000000e+00}:9.987200e-01:127>
!qalias = !quant.quantile<f8E4M3FN<-448:448>:f32, {-1.0,1.0}:0.99872:127  >
func.func @parse() -> !qalias {
  %0 = "foo"() : () -> !qalias
  return %0 : !qalias
}

// -----
// Required per-layer params specified:
//   [unsigned] storageType, expressedType, scale
// CHECK: !quant.quantile<u8:f32, {-1.000000e+00,1.000000e+00}:9.987200e-01>
!qalias = !quant.quantile<u8:f32, {-1.0,1.0}:0.99872>
func.func @parse() -> !qalias {
  %0 = "foo"() : () -> !qalias
  return %0 : !qalias
}

// -----
// Exponential scale (-)
// CHECK: !quant.quantile<u8:f32, {-1.000000e+00,1.000000e+00}:2.000000e-02>
!qalias = !quant.quantile<u8:f32, {-1.0,1.0}:2.0e-2>
func.func @parse() -> !qalias {
  %0 = "foo"() : () -> !qalias
  return %0 : !qalias
}

// -----
// Exponential scale (+)
// CHECK: !quant.quantile<u8:f32, {-1.000000e+00,1.000000e+00}:2.000000e+02>
!qalias = !quant.quantile<u8:f32, {-1.0,1.0}:2.0e+2>
func.func @parse() -> !qalias {
  %0 = "foo"() : () -> !qalias
  return %0 : !qalias
}

// -----
// Storage type: f8E5M2
// CHECK: !quant.quantile<f8E5M2:f32, {-1.000000e+00,1.000000e+00}:2.000000e+02>
!qalias = !quant.quantile<f8E5M2:f32, {-1.0,1.0}:2.0e+2>
func.func @parse() -> !qalias {
  %0 = "foo"() : () -> !qalias
  return %0 : !qalias
}

// -----
// Storage type: f8E4M3FN
// CHECK: !quant.quantile<f8E4M3FN:f32, {-1.000000e+00,1.000000e+00}:2.000000e+02>
!qalias = !quant.quantile<f8E4M3FN:f32, {-1.0,1.0}:2.0e+2>
func.func @parse() -> !qalias {
  %0 = "foo"() : () -> !qalias
  return %0 : !qalias
}

// -----
// Storage type: i16
// CHECK: !quant.quantile<i16:f32, {-1.000000e+00,1.000000e+00}:2.000000e+02>
!qalias = !quant.quantile<i16:f32, {-1.0,1.0}:2.0e+2>
func.func @parse() -> !qalias {
  %0 = "foo"() : () -> !qalias
  return %0 : !qalias
}

// -----
// Storage type: u16
// CHECK: !quant.quantile<u16:f32, {-1.000000e+00,1.000000e+00}:2.000000e+02>
!qalias = !quant.quantile<u16:f32, {-1.0,1.0}:2.0e+2>
func.func @parse() -> !qalias {
  %0 = "foo"() : () -> !qalias
  return %0 : !qalias
}

// -----
// Storage type: i32
// CHECK: !quant.quantile<i32:f32, {-1.000000e+00,1.000000e+00}:2.000000e+02>
!qalias = !quant.quantile<i32:f32, {-1.0,1.0}:2.0e+2>
func.func @parse() -> !qalias {
  %0 = "foo"() : () -> !qalias
  return %0 : !qalias
}

// -----
// Storage type: u32
// CHECK: !quant.quantile<u32:f32, {-1.000000e+00,1.000000e+00}:2.000000e+02>
!qalias = !quant.quantile<u32:f32, {-1.0,1.0}:2.0e+2>
func.func @parse() -> !qalias {
  %0 = "foo"() : () -> !qalias
  return %0 : !qalias
}

// -----
// Expressed type: f32
// CHECK: !quant.quantile<u8:f32, {-1.000000e+00,1.000000e+00}:2.000000e+02>
!qalias = !quant.quantile<u8:f32, {-1.0,1.0}:2.0e+2>
func.func @parse() -> !qalias {
  %0 = "foo"() : () -> !qalias
  return %0 : !qalias
}

// -----
// Expressed type: f32
// CHECK: !quant.quantile<u8:f32, {-1.000000e+00,1.000000e+00}:0x41646ABBA0000000:128>
!qalias = !quant.quantile<u8:f32, {-1.0,1.0}:0x41646ABBA0000000:128>
func.func @parse() -> !qalias {
  %0 = "foo"() : () -> !qalias
  return %0 : !qalias
}

// -----
// Expressed type: f16
// CHECK: !quant.quantile<u8:f16, {-1.000000e+00,1.000000e+00}:2.000000e+02>
!qalias = !quant.quantile<u8:f16, {-1.0,1.0}:2.0e+2>
func.func @parse() -> !qalias {
  %0 = "foo"() : () -> !qalias
  return %0 : !qalias
}

// -----
// Expressed type: f64
// CHECK: !quant.quantile<u8:f64, {-1.000000e+00,1.000000e+00}:2.000000e+02>
!qalias = !quant.quantile<u8:f64, {-1.0,1.0}:2.0e+2>
func.func @parse() -> !qalias {
  %0 = "foo"() : () -> !qalias
  return %0 : !qalias
}

// -----
// Expressed type: bf16
// CHECK: !quant.quantile<u8:bf16, {-1.000000e+00,1.000000e+00}:2.000000e+02>
!qalias = !quant.quantile<u8:bf16, {-1.0,1.0}:2.0e+2>
func.func @parse() -> !qalias {
  %0 = "foo"() : () -> !qalias
  return %0 : !qalias
}

// -----
// Per-axis scales and zero points (affine)
// CHECK: !quant.quantile<u8:f32:1, {-1.000000e+00,1.000000e+00}:{2.000000e+02:-120,9.987200e-01:127}>
!qalias = !quant.quantile<u8:f32:1, {-1.0,1.0}:{2.0e+2:-120,0.99872:127}>
func.func @parse() -> !qalias {
  %0 = "foo"() : () -> !qalias
  return %0 : !qalias
}

// -----
// Per-axis scales and no zero points (fixedpoint)
// CHECK: !quant.quantile<i8:f32:1, {-1.000000e+00,1.000000e+00}:{2.000000e+02,9.987200e-01}>
!qalias = !quant.quantile<i8:f32:1, {-1.0,1.0}:{2.0e+2,0.99872}>
func.func @parse() -> !qalias {
  %0 = "foo"() : () -> !qalias
  return %0 : !qalias
}

// -----
// Per-axis scales and zero points (mixed affine and fixedpoint)
// CHECK: !quant.quantile<i8:f32:1, {-1.000000e+00,1.000000e+00}:{2.000000e+02,9.987200e-01:120}>
!qalias = !quant.quantile<i8:f32:1, {-1.0,1.0}:{2.0e+2,0.99872:120}>
func.func @parse() -> !qalias {
  %0 = "foo"() : () -> !qalias
  return %0 : !qalias
}
