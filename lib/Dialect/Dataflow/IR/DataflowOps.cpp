//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "comet/Dialect/Dataflow/IR/DataflowOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/StringMap.h"

using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::comet::Dataflow;

#include "mlir/IR/Builders.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/FunctionSupport.h"
#include "mlir/IR/SymbolTable.h"

Region &DesignOp::getBodyRegion() { return getOperation()->getRegion(0); }
Block  *DesignOp::getBody() { return &getBodyRegion().front(); }

void DesignOp::build(OpBuilder &builder, OperationState &result, StringAttr name) {
  // Add an attribute for the name.
  result.addAttribute(builder.getIdentifier("name"), name);

  // Create a region and a block for the body.
  Region *bodyRegion = result.addRegion();
  Block *body = new Block();
  bodyRegion->push_back(body);
}

// computeDefOp
static ParseResult parseComputeDefOp(OpAsmParser &parser, OperationState &result) {
  using namespace mlir::function_like_impl;

  SmallVector<OpAsmParser::OperandType, 4> entryArgs;
  SmallVector<NamedAttrList, 4> argsAttrs;
  SmallVector<NamedAttrList, 4> resultAttrs;
  SmallVector<Type, 4> argTypes;
  SmallVector<Type, 4> resultTypes;
  auto &builder = parser.getBuilder();

  // Parse the name as a symbol.
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, ::mlir::SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();
  
  DictionaryAttr dictAttr;
  if (parser.parseAttribute(dictAttr, "_", result.attributes))
    return failure();
  result.attributes.assign(dictAttr.getValue().begin(),
                           dictAttr.getValue().end());


  // Parse the function signature.
  bool isVariadic = false;
  if (parseFunctionSignature(parser, /*allowVariadic*/ false, entryArgs,
                              argTypes, argsAttrs,
                              isVariadic, resultTypes, resultAttrs))
    return failure();

  auto type = builder.getFunctionType(argTypes, resultTypes);
  result.addAttribute(getTypeAttrName(), TypeAttr::get(type));

  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  addArgAndResultAttrs(builder, result, argsAttrs, resultAttrs);

  // Parse the optional function body.
  auto *body = result.addRegion();
  if (parser.parseRegion(*body, entryArgs,
                         entryArgs.empty() ? ArrayRef<Type>() : argTypes))
    return failure();
  if (body->empty())
    body->push_back(new Block());
  return success();
}

static void printComputeDefOp(OpAsmPrinter &p, ComputeDefOp op) {
  using namespace mlir::function_like_impl;

  auto typeAttr = op->getAttrOfType<TypeAttr>(ComputeDefOp::getTypeAttrName());
  FunctionType fnType = typeAttr.getValue().cast<FunctionType>();
  auto argTypes = fnType.getInputs();
  auto resultTypes = fnType.getResults();

  // Print the operation and the function name.
  auto funcName =
      op->getAttrOfType<StringAttr>(::mlir::SymbolTable::getSymbolAttrName())
          .getValue();
  p << op->getName() << ' ';
  p.printSymbolName(funcName);

  printFunctionSignature(p, op, argTypes, /*isVariadic*/ false, resultTypes);
  SmallVector<StringRef, 3> omittedAttrs({});

  printFunctionAttributes(p, op, argTypes.size(), resultTypes.size(),
                          omittedAttrs);
  Region &body = op.getBody();
  if (!body.empty())
    p.printRegion(body, /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);
}
// dataflow.memoryDef @bufferABO () -> !dataflow.memory<"bufferABO"> {}
void MemoryDefOp::build(OpBuilder &builder, OperationState &result,
                      StringAttr name) {

    using namespace mlir::function_like_impl;

    // Add an attribute for the name.
    result.addAttribute(::mlir::SymbolTable::getSymbolAttrName(), name);

    SmallVector<Type, 4> resType;
    MemoryType memType = MemoryType::get(result.getContext(), name.getValue()); 
    resType.push_back(memType);

    // Record the argument and result types as an attribute.
    auto type = builder.getFunctionType(/*arg types*/ {}, /*resultTypes*/ resType );
    result.addAttribute(getTypeAttrName(), TypeAttr::get(type));

    result.addRegion();

  // Create a region and a block for the body.
  auto *bodyRegion = result.regions[0].get();
  Block *body = new Block();
  bodyRegion->push_back(body);

}

void ComputeDefOp::build(OpBuilder &builder, OperationState &result,
                      StringAttr name) {

    using namespace mlir::function_like_impl;

    // Add an attribute for the name.
    result.addAttribute(::mlir::SymbolTable::getSymbolAttrName(), name);

    SmallVector<Type, 4> resType;
    MemoryType memType = MemoryType::get(result.getContext(), name.getValue()); 
    resType.push_back(memType);

    // Record the argument and result types as an attribute.
    auto type = builder.getFunctionType(/*arg types*/ {}, /*resultTypes*/ resType );
    result.addAttribute(getTypeAttrName(), TypeAttr::get(type));

    result.addRegion();

  // Create a region and a block for the body.
  auto *bodyRegion = result.regions[0].get();
  Block *body = new Block();
  bodyRegion->push_back(body);

}

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallSet.h"
#include <iostream>
void ComputeDefOp::build(OpBuilder &builder, OperationState &result,
                      StringAttr name, llvm::ArrayRef<mlir::AffineMap> index, llvm::ArrayRef<llvm::StringRef> iter) {

    using namespace mlir::function_like_impl;

    // Add an attribute for the name.
    result.addAttribute(::mlir::SymbolTable::getSymbolAttrName(), name);

    // Record the names of the arguments if present.
    SmallVector<Attribute, 4> ites;
    SmallVector<Attribute, 4> inds;
    std::cout << "orange" << std::endl;
    for (size_t i = 0, e = index.size(); i != e; ++i) {
      std::cout << " i = " << i << std::endl;
      AffineMap orig = index[i];
      // crazy big: //orig.dump();
      unsigned int dims = orig.getNumDims();
      std::cout << " DIMS " << dims << std::endl; 
      unsigned int syms = orig.getNumSymbols();
      std::cout << " SYMS " << dims << std::endl; 
      ArrayRef<AffineExpr> exps = orig.getResults();
      for (size_t q = 0, r = exps.size(); q != r; ++q) {
        exps[q].dump();
      }
      AffineMap map =  AffineMap::get(dims, syms, orig.getResults(), builder.getContext()); 
      AffineMapAttr mapAtt = AffineMapAttr::get(map);
      inds.push_back(mapAtt);
    }
    std::cout << "banana" << std::endl;
    std::cout << iter.size() << std::endl;
    for (size_t i = 0, e = iter.size(); i != e; ++i) {
      std::cout << " in the loop? " << std::endl;
      std::string s = iter[i].str();
      std::cout << " $$$ THE STRING $$$ " << s << std::endl;
      StringAttr str = builder.getStringAttr(s);
      ites.push_back(str);
    }
    result.addAttribute("indexing_maps", builder.getArrayAttr(inds));
    result.addAttribute("iterator_types", builder.getArrayAttr(ites));

    SmallVector<Type, 4> resType;
    ComputeType comType = ComputeType::get(result.getContext(), name.getValue()); 
    resType.push_back(comType);

    // Record the argument and result types as an attribute.
    auto type = builder.getFunctionType(/*arg types*/ {}, /*resultTypes*/ resType );
    result.addAttribute(getTypeAttrName(), TypeAttr::get(type));

    result.addRegion();

  // Create a region and a block for the body.
  auto *bodyRegion = result.regions[0].get();
  Block *body = new Block();
  bodyRegion->push_back(body);

}


// memoryDefOp
static ParseResult parseMemoryDefOp(OpAsmParser &parser, OperationState &result) {
  using namespace mlir::function_like_impl;

  SmallVector<OpAsmParser::OperandType, 4> entryArgs;
  SmallVector<NamedAttrList, 4> argsAttrs;
  SmallVector<NamedAttrList, 4> resultAttrs;
  SmallVector<Type, 4> argTypes;
  SmallVector<Type, 4> resultTypes;
  auto &builder = parser.getBuilder();

  // Parse the name as a symbol.
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, ::mlir::SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  // Parse the function signature.
  bool isVariadic = false;
  if (parseFunctionSignature(parser, /*allowVariadic*/ false, entryArgs,
                              argTypes, argsAttrs,
                              isVariadic, resultTypes, resultAttrs))
    return failure();

  auto type = builder.getFunctionType(argTypes, resultTypes);
  result.addAttribute(getTypeAttrName(), TypeAttr::get(type));

  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  assert(argsAttrs.size() == argTypes.size());
  assert(resultAttrs.size() == resultTypes.size());

  addArgAndResultAttrs(builder, result, argsAttrs, resultAttrs);

  auto *body = result.addRegion();
  if (parser.parseRegion(*body, entryArgs,
                         entryArgs.empty() ? ArrayRef<Type>() : argTypes))
    return failure();
  if (body->empty())
    body->push_back(new Block());
  return success();
}

static void printMemoryDefOp(OpAsmPrinter &p, MemoryDefOp op) {
  using namespace mlir::function_like_impl;

  auto typeAttr = op->getAttrOfType<TypeAttr>(MemoryDefOp::getTypeAttrName());
  FunctionType fnType = typeAttr.getValue().cast<FunctionType>();
  auto argTypes = fnType.getInputs();
  auto resultTypes = fnType.getResults();

  auto funcName =
      op->getAttrOfType<StringAttr>(::mlir::SymbolTable::getSymbolAttrName())
          .getValue();
  p << op->getName() << ' ';
  p.printSymbolName(funcName);

  printFunctionSignature(p, op, argTypes, /*isVariadic*/ false, resultTypes);
  SmallVector<StringRef, 3> omittedAttrs({});

  printFunctionAttributes(p, op, argTypes.size(), resultTypes.size(),
                          omittedAttrs);
  Region &body = op.getBody();
  if (!body.empty())
    p.printRegion(body, /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);
}


#define GET_OP_CLASSES
#include "comet/Dialect/Dataflow/IR/DataflowOps.cpp.inc"
