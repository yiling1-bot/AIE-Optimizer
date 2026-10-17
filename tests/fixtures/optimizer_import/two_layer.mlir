// Public, synthetic, non-square example. Argument names/order are intentional.
#a = affine_map<(m, n, k) -> (m, k)>
#b = affine_map<(m, n, k) -> (k, n)>
#c = affine_map<(m, n, k) -> (m, n)>
#id = affine_map<(m, n) -> (m, n)>
module {
  func.func @network(%second: tensor<24x16xi8>, %samples: tensor<8x16xi8>, %first: tensor<16x24xi8>) -> tensor<8x16xi8> {
    %zero = arith.constant 0 : i32
    %scratch0 = tensor.empty() : tensor<8x24xi32>
    %init0 = linalg.fill ins(%zero : i32) outs(%scratch0 : tensor<8x24xi32>) -> tensor<8x24xi32>
    %acc0 = linalg.generic {indexing_maps = [#a, #b, #c], iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%samples, %first : tensor<8x16xi8>, tensor<16x24xi8>) outs(%init0 : tensor<8x24xi32>) {
      ^bb0(%x: i8, %w: i8, %a: i32):
        %xx = arith.extsi %x : i8 to i32
        %ww = arith.extsi %w : i8 to i32
        %p = arith.muli %xx, %ww : i32
        %s = arith.addi %a, %p : i32
        linalg.yield %s : i32
    } -> tensor<8x24xi32>
    %hidden_space = tensor.empty() : tensor<8x24xi8>
    %hidden = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
      ins(%acc0 : tensor<8x24xi32>) outs(%hidden_space : tensor<8x24xi8>) {
      ^bb0(%a: i32, %unused: i8):
        %lo = arith.constant -128 : i32
        %hi = arith.constant 127 : i32
        %lower = arith.maxsi %a, %lo : i32
        %upper = arith.minsi %lower, %hi : i32
        %q = arith.trunci %upper : i32 to i8
        linalg.yield %q : i8
    } -> tensor<8x24xi8>
    %h32 = arith.extsi %hidden : tensor<8x24xi8> to tensor<8x24xi32>
    %w32 = arith.extsi %second : tensor<24x16xi8> to tensor<24x16xi32>
    %scratch1 = tensor.empty() : tensor<8x16xi32>
    %init1 = linalg.fill ins(%zero : i32) outs(%scratch1 : tensor<8x16xi32>) -> tensor<8x16xi32>
    %acc1 = linalg.matmul ins(%h32, %w32 : tensor<8x24xi32>, tensor<24x16xi32>) outs(%init1 : tensor<8x16xi32>) -> tensor<8x16xi32>
    %low = arith.constant dense<-128> : tensor<8x16xi32>
    %high = arith.constant dense<127> : tensor<8x16xi32>
    %upper1 = arith.minsi %high, %acc1 : tensor<8x16xi32>
    %lower1 = arith.maxsi %upper1, %low : tensor<8x16xi32>
    %result = arith.trunci %lower1 : tensor<8x16xi32> to tensor<8x16xi8>
    return %result : tensor<8x16xi8>
  }
}
