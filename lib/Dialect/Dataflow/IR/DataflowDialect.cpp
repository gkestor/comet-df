//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "comet/Dialect/Dataflow/IR/DataflowDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "comet/Dialect/Dataflow/IR/DataflowOps.h"
#include "comet/Dialect/Dataflow/IR/DataflowTypes.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::comet;
using namespace mlir::comet::Dataflow;

#include "comet/Dialect/Dataflow/IR/DataflowDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Dialect Interfaces
//===----------------------------------------------------------------------===//

namespace {
struct DataflowInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;
  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       BlockAndValueMapping &valueMapping) const final {
    return true;
  }
  bool isLegalToInline(Operation *, Region *, bool wouldBeCloned,
                       BlockAndValueMapping &) const final {
    return true;
  }
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Tablegen Type Definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "comet/Dialect/Dataflow/IR/DataflowTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// Dialect initialize method.
//===----------------------------------------------------------------------===//

void DataflowDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "comet/Dialect/Dataflow/IR/DataflowOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "comet/Dialect/Dataflow/IR/DataflowTypes.cpp.inc"
      >();
  addInterfaces<DataflowInlinerInterface>();
}

//===----------------------------------------------------------------------===//
// Type-related Dialect methods.
//===----------------------------------------------------------------------===//

Type DataflowDialect::parseType(DialectAsmParser &parser) const {
  StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return Type();
  Type type;
  if (generatedTypeParser(getContext(), parser, keyword, type).hasValue())
    return type;

  parser.emitError(parser.getNameLoc(), "invalid 'dataflow' type: `")
      << keyword << "'";
  return Type();
}

void DataflowDialect::printType(Type type, DialectAsmPrinter &printer) const {
  if (failed(generatedTypePrinter(type, printer)))
    llvm_unreachable("unknown 'dataflow' type");
}


//===----------------------------------------------------------------------===//
// Dialect-level verifiers.
//===----------------------------------------------------------------------===//

LogicalResult DataflowDialect::verifyRegionArgAttribute(Operation *op,
                                                     unsigned regionIndex,
                                                     unsigned argIndex,
                                                     NamedAttribute namedAttr) {
  if (namedAttr.first == "dataflow.type_bound") {
    auto func = dyn_cast<FuncOp>(op);
    if (!func)
      return op->emitError() << "'dataflow.type_bound' must be attached to a func";
    TypeAttr attr = namedAttr.second.dyn_cast<TypeAttr>();
    if (!attr)
      return op->emitError() << "'dataflow.type_bound' must be TypeAttr";
    return success();
  }

  return op->emitError() << "unknown region arg attribute '" << namedAttr.first
                         << "'";
}

