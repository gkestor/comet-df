//This output is produced from linalg_mult.mlir

#map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
builtin.module  {
  builtin.func @matmul(%arg0: memref<10x10xf32>, %arg1: memref<10x10xf32>, %arg2: memref<10x10xf32>) {
    linalg.generic {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : memref<10x10xf32>, memref<10x10xf32>) outs(%arg2 : memref<10x10xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):  // no predecessors
      %0 = mulf %arg3, %arg4 : f32
      %1 = addf %arg5, %0 : f32
      linalg.yield %1 : f32
    }
    return
  }
}