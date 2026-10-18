module {
  func.func private @stage(%x: tensor<4x8xi8>, %w: tensor<8x8xi8>) -> tensor<4x8xi8> {
    %z = arith.constant dense<0> : tensor<4x8xi32>
    %lo = arith.constant dense<-128> : tensor<4x8xi32>
    %hi = arith.constant dense<127> : tensor<4x8xi32>
    %p = linalg.matmul ins(%x, %w : tensor<4x8xi8>, tensor<8x8xi8>) outs(%z : tensor<4x8xi32>) -> tensor<4x8xi32>
    %a = arith.maxsi %p, %lo : tensor<4x8xi32>
    %b = arith.minsi %a, %hi : tensor<4x8xi32>
    %y = arith.trunci %b : tensor<4x8xi32> to tensor<4x8xi8>
    return %y : tensor<4x8xi8>
  }
  func.func @three_layer(%x: tensor<4x8xi8>, %a: tensor<8x8xi8>, %b: tensor<8x8xi8>, %c: tensor<8x8xi8>) -> tensor<4x8xi8> {
    %h0 = func.call @stage(%x, %a) : (tensor<4x8xi8>, tensor<8x8xi8>) -> tensor<4x8xi8>
    %h1 = func.call @stage(%h0, %b) : (tensor<4x8xi8>, tensor<8x8xi8>) -> tensor<4x8xi8>
    %y = func.call @stage(%h1, %c) : (tensor<4x8xi8>, tensor<8x8xi8>) -> tensor<4x8xi8>
    return %y : tensor<4x8xi8>
  }
}
