module {
  func.func private @shift(%x: tensor<3x4xi8>) -> tensor<3x4xi8> {
    %one = arith.constant dense<1> : tensor<3x4xi8>
    %a = arith.addi %x, %one : tensor<3x4xi8>
    return %a : tensor<3x4xi8>
  }
  func.func @branched(%input: tensor<3x4xi8>) -> (tensor<4x3xi8>, tensor<3x4xi8>) {
    %a = func.call @shift(%input) : (tensor<3x4xi8>) -> tensor<3x4xi8>
    %b = arith.muli %a, %input : tensor<3x4xi8>
    %space = tensor.empty() : tensor<4x3xi8>
    %t = linalg.transpose ins(%b : tensor<3x4xi8>) outs(%space : tensor<4x3xi8>) permutation = [1, 0]
    %even = tensor.extract_slice %a[0, 0][3, 2][1, 2] : tensor<3x4xi8> to tensor<3x2xi8>
    %odd = tensor.extract_slice %a[0, 1][3, 2][1, 2] : tensor<3x4xi8> to tensor<3x2xi8>
    %join = tensor.concat dim(1) %even, %odd : (tensor<3x2xi8>, tensor<3x2xi8>) -> tensor<3x4xi8>
    return %t, %join : tensor<4x3xi8>, tensor<3x4xi8>
  }
}
