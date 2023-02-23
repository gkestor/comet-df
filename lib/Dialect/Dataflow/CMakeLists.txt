add_llvm_library(COMETDataflow
  IR/DataflowDialect.cpp
  IR/DataflowOps.cpp
  IR/DataflowTypes.cpp

  Conversion/LinalgToDataflow.cpp  

  ADDITIONAL_HEADER_DIRS
  ${COMET_MAIN_INCLUDE_DIR}/comet/Dialect/Dataflow

  DEPENDS
  COMETDataflowOpsIncGen
  COMETDataflowTypesIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRControlFlowInterfaces
  MLIRSideEffectInterfaces
  MLIRPass
  MLIRLinalg
)
