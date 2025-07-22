def QuantileFloatTypeInterface : TypeInterface<"QuantileFloatTypeInterface"> {
  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<
      "int", "getStorageBitWidth", (ins), [{
        return $_self.getIntOrFloatBitWidth();
      }]
    >
  ];
}
