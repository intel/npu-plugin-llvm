// RUN: mlir-opt --split-input-file --test-bufferize -canonicalize -cse %s | FileCheck %s

// CHECK-LABEL: func.func @custom_dialect_op(
// CHECK-SAME:    %[[ARG:.*]]: !test.test_tensor<[32, 64], f64>
// CHECK-SAME:  ) -> !test.test_tensor<[32, 64], f64> {
// CHECK:         %[[MEMREF:.*]] = bufferization.to_memref %[[ARG]] : !test.test_memref<[32, 64], f64>
// CHECK:         %[[DUMMY:.*]] = "test.dummy_memref_op"(%[[MEMREF]]) : (!test.test_memref<[32, 64], f64>)
// CHECK-SAME:      -> !test.test_memref<[32, 64], f64>
// CHECK:         %[[OUT:.*]] = bufferization.to_tensor %[[DUMMY]] : !test.test_memref<[32, 64], f64>
// CHECK:         return %[[OUT]] : !test.test_tensor<[32, 64], f64>
// CHECK:       }
func.func @custom_dialect_op(%arg: !test.test_tensor<[32, 64], f64>) -> !test.test_tensor<[32, 64], f64> {
  %out = "test.dummy_tensor_op"(%arg) : (!test.test_tensor<[32, 64], f64>) -> !test.test_tensor<[32, 64], f64>
  return %out : !test.test_tensor<[32, 64], f64>
}
