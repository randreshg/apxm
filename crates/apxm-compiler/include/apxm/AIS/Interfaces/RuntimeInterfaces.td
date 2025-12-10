//===- RuntimeInterfaces.td - AIS Runtime Interfaces -------*- tablegen -*-===//
//
// Part of the A-PXM project, under the Apache License v2.0.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#ifndef APXM_AIS_RUNTIME_INTERFACES
#define APXM_AIS_RUNTIME_INTERFACES

include "mlir/IR/OpBase.td"

def AISRuntimeCallableOpInterface : OpInterface<"RuntimeCallableOpInterface"> {
  let cppNamespace = "::mlir::ais";
  let description = [{
    Interface implemented by AIS operations that can be lowered into
    concrete runtime calls.  It exposes the entry point symbol along with
    the operands/results that need to be forwarded to the runtime.
  }];
  let methods = [
    InterfaceMethod<
      /*desc=*/"Return the ais-runtime symbol invoked to execute this op.",
      /*retTy=*/"llvm::StringRef",
      /*methodName=*/"getRuntimeEntryPoint",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/"Return the operands forwarded to the runtime call.",
      /*retTy=*/"::mlir::OperandRange",
      /*methodName=*/"getRuntimeOperands",
      /*args=*/(ins),
      /*methodBody=*/[{
        return $_op->getOperands();
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Return the expected result types of the runtime call.",
      /*retTy=*/"::mlir::TypeRange",
      /*methodName=*/"getRuntimeResultTypes",
      /*args=*/(ins),
      /*methodBody=*/[{
        return $_op->getResultTypes();
      }]
    >
  ];
}

#endif // APXM_AIS_RUNTIME_INTERFACES
