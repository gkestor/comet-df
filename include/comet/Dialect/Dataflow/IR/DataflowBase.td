//===-------------------------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_BASE
#define DATAFLOW_BASE

include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"

def Dataflow_Dialect : Dialect {
  let name = "dataflow";
  let cppNamespace = "::mlir::comet::Dataflow";
  let description = [{
    Top-level dialect for interfacing PyDataflow and MLIR.

    This dialect maintains a fairly isomorphic representation with DataflowScript.

    TODO: Add more detail here.
  }];

  let hasRegionArgAttrVerify = 1;
}

class DataflowOpTrait<string name> : OpTrait, NativeTrait<"", ""> {
  let trait = name;
  let cppNamespace = "::mlir::comet::Dataflow::OpTrait";
}

def HasValueSemantics : DataflowOpTrait<"HasValueSemantics">;
def IsTrailingUnderscoreInplaceVariant
  : DataflowOpTrait<"IsTrailingUnderscoreInplaceVariant">;

#endif // DATAFLOW_BASE
