//===-------------------------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_TYPES
#define DATAFLOW_TYPES

include "comet/Dialect/Dataflow/IR/DataflowBase.td"


class Dataflow_Type<string name, string typeMnemonic,
                 string baseCppClass = "::mlir::Type">
    : TypeDef<Dataflow_Dialect, name, [], baseCppClass> {
  let mnemonic = typeMnemonic;
}


def Dataflow_ComputeType : Dataflow_Type<"Compute", "compute"> {
  let summary = "dataflow.compute";
  let description = [{
    Represents an instance of a `dataflow.computeDef` with the given `className`.
  }];
  let parameters = (ins StringRefParameter<"class name">:$className);

  let printer = [{
    $_printer << "compute<\"";
    llvm::printEscapedString(getImpl()->className, $_printer.getStream());
    $_printer << "\">";
  }];

  let parser = [{
    if (parser.parseLess())
      return Type();
    StringRef className;
    if ($_parser.parseOptionalString(&className))
      return Type();
    if ($_parser.parseGreater())
      return Type();
    return get($_ctxt, className);
  }];
}


def Dataflow_MemoryType : Dataflow_Type<"Memory", "memory"> {
  let summary = "dataflow.memory";
  let description = [{
    Represents an instance of a `dataflow.memoryDef` with the given `className`.
  }];
  let parameters = (ins StringRefParameter<"class name">:$className);

  let printer = [{
    $_printer << "memory<\"";
    llvm::printEscapedString(getImpl()->className, $_printer.getStream());
    $_printer << "\">";
  }];

  let parser = [{
    if (parser.parseLess())
      return Type();
    StringRef className;
    if ($_parser.parseOptionalString(&className))
      return Type();
    if ($_parser.parseGreater())
      return Type();
    return get($_ctxt, className);
  }];
}

def Dataflow_VecType : Dataflow_Type<"Vec", "vec"> {
  let summary = "Dataflow VecType";
  let description = [{
    Just indicates some vector that is eventually mapped to a memory. 
  }];
}

def AnyDataflowType : AnyTypeOf<[
  Dataflow_MemoryType,
  Dataflow_ComputeType,
], "Any type that is legal">;

#endif // DATAFLOW_TYPES
