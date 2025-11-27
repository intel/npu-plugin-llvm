#ifndef MLIR_IR_QUANTIZATIONINTERFACE
#define MLIR_IR_QUANTIZATIONINTERFACE

include "mlir/IR/OpBase.td"

def QuantizationInterface : TypeInterface<"QuantizationInterface"> {
  let description = [{
    Interface for types that can be used as quantile storage types.
    This interface provides methods to determine storage characteristics
    like width and signedness for quantization purposes.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<[{
      Get the storage type width in bits.
      Returns the number of bits used to store values of this type.
    }],
    "unsigned", "getStorageWidth", (ins)>,
    
    InterfaceMethod<[{
      Check if the storage type is signed.
      Returns true if the type represents signed values, false for unsigned.
    }],
    "bool", "isStorageSigned", (ins)>,

    InterfaceMethod<[{
      Get the default minimum value for the storage type.
    }],
    "int64_t", "getDefaultMinimum", (ins "bool":$isSigned, "unsigned":$integralWidth)>,

    InterfaceMethod<[{
      Get the default maximum value for the storage type.
    }],
    "int64_t", "getDefaultMaximum", (ins "bool":$isSigned, "unsigned":$integralWidth)>,

    InterfaceMethod<[{
      Get the name of the storage type.
    }],
    "std::string", "printStorageType", (ins "bool":$isSigned, "unsigned":$storageWidth)>
  ];

}

#endif // MLIR_IR_QUANTIZATIONINTERFACE
