//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "comet/Dialect/Dataflow/IR/DataflowOps.h"
#include "comet/Dialect/Dataflow/IR/DataflowDialect.h"
#include "comet/Dialect/Dataflow/Passes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"
#include <iostream>

#include "mlir/IR/ImplicitLocOpBuilder.h"

using namespace mlir;
using namespace mlir::comet::Dataflow;

/// Given a FIRRTL type, return the corresponding type for the HW dialect.
/// This returns a null type if it cannot be lowered.
static Type lowerType(Type type)
{
  return type;
}

// -----------------------------------------------------------------------------
// The pass
// -----------------------------------------------------------------------------

namespace
{
  struct LowerLinalgToDataflow
  {
    LowerLinalgToDataflow(ModuleOp modOp, FuncOp funcOp) : modOp(modOp),
                                                           builder(funcOp.getLoc(), funcOp.getContext()) {}
    void visitGeneric(linalg::GenericOp op);
    void visitFunc(FuncOp funcOp);

  private:
    // linalg::GenericOp genOp;
    // DesignOp desOp;
    ModuleOp modOp;
    ImplicitLocOpBuilder builder;
  };
} // namespace

// namespace
// {
//   class ConvertLinalgToDataflow
//      : public mlir::PassWrapper<ConvertLinalgToDataflow, mlir::OperationPass>

//   // class ConvertLinalgToDataflow
//   //     : public ConvertLinalgToDataflowBase<ConvertLinalgToDataflow>
//   {
//   public:
//     void getDependentDialects(DialectRegistry &registry) const override
//     {
//       registry.insert<linalg::LinalgDialect>();
//       registry.insert<mlir::comet::Dataflow::DataflowDialect>();
//       registry.insert<math::MathDialect>();
//       registry.insert<StandardOpsDialect>();
//       registry.insert<tensor::TensorDialect>();
//     }

//     void runOnOperation() override
//     {
//       MLIRContext *context = &getContext();
//       ConversionTarget target(*context);
//       target.addLegalDialect<linalg::LinalgDialect, StandardOpsDialect,
//                              math::MathDialect, tensor::TensorDialect,
//                              mlir::comet::Dataflow::DataflowDialect>();

//       std::cout << "test" << std::endl;
//       TypeConverter typeConverter;
//       typeConverter.addConversion([](Type type)
//                                   { return type; });
//       auto *body = getOperation().getBody();

//       SmallVector<Operation *, 16> opsToRemove;

//       ModuleOp modOp = getOperation();
//       for (auto &op : body->getOperations())
//       {
//         op.dump();
//         if (auto funcOp = dyn_cast<FuncOp>(op))
//         {
//           LowerLinalgToDataflow(modOp, funcOp).visitFunc(funcOp);
//         }
//       }
//     }
//   };
// } // namespace

namespace
{
  struct ConvertLinalgToDataflowPass
      : public PassWrapper<ConvertLinalgToDataflowPass, OperationPass<ModuleOp>>
  {
    void getDependentDialects(DialectRegistry &registry) const override
    {
      registry.insert<linalg::LinalgDialect>();
      registry.insert<mlir::comet::Dataflow::DataflowDialect>();
      registry.insert<math::MathDialect>();
      registry.insert<StandardOpsDialect>();
      registry.insert<tensor::TensorDialect>();
    }
    void runOnOperation() final;
  };
} // end anonymous namespace

void ConvertLinalgToDataflowPass::runOnOperation()
{
  MLIRContext *context = &getContext();
  ConversionTarget target(*context);
  target.addLegalDialect<linalg::LinalgDialect, StandardOpsDialect,
                         math::MathDialect, tensor::TensorDialect,
                         mlir::comet::Dataflow::DataflowDialect>();

  std::cout << "test" << std::endl;
  TypeConverter typeConverter;
  typeConverter.addConversion([](Type type)
                              { return type; });
  auto *body = getOperation().getBody();

  SmallVector<Operation *, 16> opsToRemove;

  ModuleOp modOp = getOperation();
  for (auto &op : body->getOperations())
  {
    op.dump();
    if (auto funcOp = dyn_cast<FuncOp>(op))
    {
      LowerLinalgToDataflow(modOp, funcOp).visitFunc(funcOp);
    }
  }
}

// There is a lot of debugging collateral in here.
void LowerLinalgToDataflow::visitFunc(FuncOp funcOp)
{
  Location loc = funcOp->getLoc();
  loc.dump();
  builder.setInsertionPoint(funcOp);
  StringAttr name = builder.getStringAttr("des_repl");
  auto design =
      builder.create<DesignOp>(loc, name);
  modOp.dump();

  builder.setInsertionPointToStart(design.getBody());
  StringAttr nameMem = builder.getStringAttr("mem");
  StringAttr nameCom = builder.getStringAttr("comp");
  auto memOp = builder.create<MemoryDefOp>(loc, nameMem);

  ArrayRef<AffineMap> ind;
  ArrayRef<StringRef> ite;
  for (auto &block : funcOp)
  {
    block.dump();
    for (auto &op : block)
    {
      if (auto genOp = dyn_cast<linalg::GenericOp>(op))
      {
        op.dump();
        std::vector<AffineMap> index;
        for (auto y = genOp.indexing_mapsAttr().begin();
             y != genOp.indexing_mapsAttr().end(); ++y)
        {
          Attribute yy = *y;
          AffineMapAttr yyy = yy.dyn_cast<AffineMapAttr>();
          AffineMap ref = yyy.getValue();
          ref.dump();
          index.push_back(ref);
        }
        std::vector<StringRef> iter;
        for (auto y = genOp.iterator_typesAttr().begin();
             y != genOp.iterator_typesAttr().end(); ++y)
        {
          Attribute yy = *y;
          StringAttr yyy = yy.dyn_cast<StringAttr>();
          StringRef ref = yyy.getValue();
          std::cout << "string is " << ref.str() << std::endl;
          iter.push_back(ref);
        }
        auto comOp = builder.create<ComputeDefOp>(loc, nameCom, index, iter);
        BlockAndValueMapping bvm;
        genOp.region().cloneInto(&comOp.getRegion(), bvm);

        auto blockArgs = genOp.region().front().getArguments();
        for (auto ba : blockArgs)
        {
          ba.dump();
        }
        auto blockArgTypes = genOp.region().front().getArgumentTypes();
        for (auto ba : blockArgTypes)
        {
          ba.dump();
        }
        comOp.getRegion().front().erase();
        comOp.getRegion().front().addArguments(genOp.region().front().getArgumentTypes());
        linalg::YieldOp to_erase;
        for (auto &innerOp : comOp.getRegion().front())
        {
          if (auto yieldOp = dyn_cast<linalg::YieldOp>(innerOp))
          {
            to_erase = yieldOp;
            builder.setInsertionPointAfter(yieldOp);
            auto dfyield = builder.create<mlir::comet::Dataflow::YieldOp>(loc, yieldOp.getOperands());
            yieldOp.dump();
            dfyield.dump();
            comOp.dump();
            builder.setInsertionPointAfter(dfyield);
          }
        }
        to_erase.erase();
      }
    }
  }

  design.dump();
  modOp.dump();
  return;
}

void LowerLinalgToDataflow::visitGeneric(linalg::GenericOp op)
{
}

std::unique_ptr<mlir::Pass> mlir::comet::Dataflow::createConvertLinalgToDataflowPass()
{
  return std::make_unique<ConvertLinalgToDataflowPass>();
}
