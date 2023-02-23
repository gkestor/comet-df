//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef COMET_DIALECT_DATAFLOW_IR_DATAFLOWTYPES_H
#define COMET_DIALECT_DATAFLOW_IR_DATAFLOWTYPES_H

#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
namespace comet {
namespace Dataflow {

} // namespace Dataflow
} // namespace comet
} // namespace mlir

#define GET_TYPEDEF_CLASSES
#include "comet/Dialect/Dataflow/IR/DataflowTypes.h.inc"

namespace mlir {
namespace comet {
namespace Dataflow {

} // namespace Dataflow
} // namespace COMET
} // namespace mlir

#endif // COMET_DIALECT_DATAFLOW_IR_DATAFLOWTYPES_H
