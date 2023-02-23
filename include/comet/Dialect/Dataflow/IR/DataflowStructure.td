
def DesignOp : Dataflow_Op<"design",
      [IsolatedFromAbove, SymbolTable, SingleBlock, NoTerminator,
       NoRegionArguments]> {
  let summary = "Dataflow Design";
  let description = [{ The design op is meant to hold compute definitions, 
                       memory definitions, instance of each of those, and 
                       mappings between buffers and compute instance inputs/outputs. }];
  let arguments = (ins StrAttr:$name);
  let results = (outs);
  let regions = (region SizedRegion<1>:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "StringAttr":$name)>
  ];

  let extraClassDeclaration = [{
    OpBuilder getBodyBuilder() {
      assert(!body().empty() && "Unexpected empty 'body' region.");
      Block &bodyBlock = body().front();
      return OpBuilder::atBlockEnd(&bodyBlock);
    }

    /// Return body of this design.
    Region &getBodyRegion();
    Block *getBody();
  }];

  let assemblyFormat = "$name $body attr-dict";
  // TODO: write the verifier. 
  // let verifier = "return ::verifyDesignOp(*this);";
}

def MemoryDefOp : Dataflow_Op<"memoryDef",
      [IsolatedFromAbove, FunctionLike, Symbol, SingleBlock, NoTerminator,
       HasParent<"DesignOp">]> {
  let summary = "Dataflow Memory";
  let description = [{ This op is meant to describe an abstract memory unit. 
                        Currently unclear if this needs to be its own definition
                        or if the type system can just absorb the memory unit info. 
                        It's FunctionLike because it describes a memory unit and 
                        this op returns an instance of a memory unit of that type. 
                        In the future, this op may take arguments.}];
  
  let arguments = (ins SymbolNameAttr:$sym_name);
  let regions = (region SizedRegion<1>:$body);

  let results = (outs);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "StringAttr":$name)>
  ];

  let extraClassDeclaration = [{
    using FunctionLike::end;
    using FunctionLike::front;
    using FunctionLike::getBody;

    Block *getBodyBlock() { return &body().front(); }

    OpBuilder getBodyBuilder() {
      assert(!body().empty() && "Unexpected empty 'body' region.");
      Block &bodyBlock = body().front();
      return OpBuilder::atBlockEnd(&bodyBlock);
    }

  private:
    friend class mlir::OpTrait::FunctionLike<MemoryDefOp>;

    unsigned getNumFuncArguments() { return getBodyBlock()->getNumArguments(); }
    unsigned getNumFuncResults() { return getType().getResults().size(); }

    LogicalResult verifyType() {
      auto type = getTypeAttr().getValue();
      if (!type.isa<FunctionType>())
        return emitOpError("requires '" + getTypeAttrName() +
                           "' attribute of function type");
      return success();
    }
  public:

  }];

  let printer = "return ::print$cppClass(p, *this);";
  let parser = "return ::parse$cppClass(parser, result);";
  // TODO: Write verifier.
  // let verifier = "return ::verifyMemoryDefOp(*this);";
}

def ComputeDefOp : Dataflow_Op<"computeDef",
      [IsolatedFromAbove, FunctionLike, Symbol, SingleBlock, SingleBlockImplicitTerminator<"YieldOp">,
       HasParent<"DesignOp">]> {
  let summary = "Dataflow Compute";
  let description = [{ The compute op is almost the same as a memory op, but it is meant to 
                       describe some compute unit. }];
  
  let arguments = (ins SymbolNameAttr:$sym_name,
                       AffineMapArrayAttr:$indexing_maps,
                       ArrayAttr:$iterator_types);
  let regions = (region SizedRegion<1>:$body);

  let results = (outs);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "StringAttr":$name)>,
    OpBuilder<(ins "StringAttr":$name, "ArrayRef<AffineMap>":$indexingMaps, "ArrayRef<StringRef>":$iteratorTypes)>
  ];

  let extraClassDeclaration = [{
    using FunctionLike::end;
    using FunctionLike::front;
    using FunctionLike::getBody;

    Block *getBodyBlock() { return &body().front(); }

    OpBuilder getBodyBuilder() {
      assert(!body().empty() && "Unexpected empty 'body' region.");
      Block &bodyBlock = body().front();
      return OpBuilder::atBlockEnd(&bodyBlock);
    }

  private:
    friend class mlir::OpTrait::FunctionLike<ComputeDefOp>;

    unsigned getNumFuncArguments() { return getBodyBlock()->getNumArguments(); }
    unsigned getNumFuncResults() { return getType().getResults().size(); }

    LogicalResult verifyType() {
      auto type = getTypeAttr().getValue();
      if (!type.isa<FunctionType>())
        return emitOpError("requires '" + getTypeAttrName() +
                           "' attribute of function type");
      return success();
    }
  public:

  }];

  let printer = "return ::print$cppClass(p, *this);";
  let parser = "return ::parse$cppClass(parser, result);";
  // TODO: Write verifier. 
  // let verifier = "return ::verifyComputeDefOp(*this);";
}

def Dataflow_YieldOp : Dataflow_Op<"yield", [NoSideEffect, ReturnLike, Terminator]>,
    Arguments<(ins Variadic<AnyType>:$values)> {
  let summary = "Dataflow Yield";
  let description = [{ Analogous to Linalg YieldOp, used in ComputeDef ops. }];
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
  let assemblyFormat = [{$values `:` type($values)  attr-dict}];
}

def InstanceMemOp : Dataflow_Op<"instanceMem",
                       [HasParent<"DesignOp">, Symbol]> {
  let summary = "Create an instance of a memoryDef";
  let description = [{ This creates an instance of a memory unit from some memory def.
                       The current implementation is only a first pass that borrows heavily 
                       from the HW dialect instance op in CIRCT. }];

  let arguments = (ins StrAttr:$instanceName,
                       FlatSymbolRefAttr:$moduleName,
                       Variadic<AnyType>:$inputs,
                       OptionalAttr<DictionaryAttr>:$parameters,
                       OptionalAttr<SymbolNameAttr>:$sym_name);
  let results = (outs Dataflow_MemoryType:$result);

  let extraClassDeclaration = [{
    // Return the name of the specified result or empty string if it cannot be
    // determined.
    StringAttr getResultName(size_t i);

    /// Lookup the module or extmodule for the symbol.  This returns null on
    /// invalid IR.
    Operation *getReferencedModule();

    /// Get the instances's name as StringAttr.
    StringAttr getNameAttr() {
      return (*this)->getAttrOfType<StringAttr>("instanceName");
    }

    /// Set the instance's name as StringAttr.
    void setNameAttr(StringAttr name) {
      (*this)->setAttr("instanceName", name);
    }

    /// Get the instances's name.
    StringRef getName() {
      return getNameAttr().getValue();
    }

    /// Set the instance's name.
    void setName(StringRef name) {
      setNameAttr(StringAttr::get(getContext(), name));
    }

    //===------------------------------------------------------------------===//
    // SymbolOpInterface Methods
    //===------------------------------------------------------------------===//

    /// An InstanceOp may optionally define a symbol.
    bool isOptionalSymbol() { return true; }

  }];

  /// sym keyword for optional symbol simplifies parsing
  // TODO: Not the final format for this op.
  let assemblyFormat = [{
    $instanceName (`sym` $sym_name^)? $moduleName `(` $inputs `)` attr-dict
      `:` functional-type($inputs, $result)
  }];
}

def InstanceComOp : Dataflow_Op<"instanceCom",
                       [HasParent<"DesignOp">, Symbol]> {
  let summary = "Create an instance of a computeDef";
  let description = [{Same as InstanceMemOp.}];

  let arguments = (ins StrAttr:$instanceName,
                       FlatSymbolRefAttr:$moduleName,
                       Variadic<AnyType>:$inputs,
                       OptionalAttr<DictionaryAttr>:$parameters,
                       OptionalAttr<SymbolNameAttr>:$sym_name);
  let results = (outs Dataflow_ComputeType:$result);

  let extraClassDeclaration = [{
    // Return the name of the specified result or empty string if it cannot be
    // determined.
    StringAttr getResultName(size_t i);

    /// Lookup the module or extmodule for the symbol.  This returns null on
    /// invalid IR.
    Operation *getReferencedModule();

    /// Get the instances's name as StringAttr.
    StringAttr getNameAttr() {
      return (*this)->getAttrOfType<StringAttr>("instanceName");
    }

    /// Set the instance's name as StringAttr.
    void setNameAttr(StringAttr name) {
      (*this)->setAttr("instanceName", name);
    }

    /// Get the instances's name.
    StringRef getName() {
      return getNameAttr().getValue();
    }

    /// Set the instance's name.
    void setName(StringRef name) {
      setNameAttr(StringAttr::get(getContext(), name));
    }

    //===------------------------------------------------------------------===//
    // SymbolOpInterface Methods
    //===------------------------------------------------------------------===//

    /// An InstanceOp may optionally define a symbol.
    bool isOptionalSymbol() { return true; }

  }];

  /// sym keyword for optional symbol simplifies parsing
  let assemblyFormat = [{
    $instanceName (`sym` $sym_name^)? $moduleName `(` $inputs `)` attr-dict
      `:` functional-type($inputs, $result)
  }];
}

def MapOp : Dataflow_Op<"map"> {
  let summary = "Maps a memory to a compute unit's buffer";
  let description = [{
    ```
      dataflow.map %computeVec, %buffer : t1, t2
    ```
    }];
  // dest is the computeVec, and src is the buffer. 
  let arguments = (ins SymbolRefAttr:$dest, FlatSymbolRefAttr:$src);
  let results = (outs);

  let assemblyFormat =
    "$dest `,` $src  attr-dict ";
}

def AddComputeVec : Dataflow_Op<"compute.vec",
                       [HasParent<"ComputeDefOp">, Symbol]> {
  let summary = "Create a vec in the compute def";
  let description = [{This is how inputs and outputs of compute units are described.}];

  let arguments = (ins SymbolNameAttr:$sym_name);
  let results = (outs Dataflow_VecType:$result);

  let extraClassDeclaration = [{
    // Return the name of the specified result or empty string if it cannot be
    // determined.
    StringAttr getResultName(size_t i);
    bool isOptionalSymbol() { return false; }

  }];

  // TODO: probably not the final implementation. 
  let assemblyFormat = [{
    $sym_name attr-dict `:` type($result)
  }];
}

