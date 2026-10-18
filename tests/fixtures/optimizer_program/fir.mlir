module {
  func.func @fir(%x: tensor<10xi16>) -> tensor<8xi16> {
    %a = tensor.extract_slice %x[0][8][1] : tensor<10xi16> to tensor<8xi16>
    %b = tensor.extract_slice %x[1][8][1] : tensor<10xi16> to tensor<8xi16>
    %c = tensor.extract_slice %x[2][8][1] : tensor<10xi16> to tensor<8xi16>
    %two = arith.constant dense<2> : tensor<8xi16>
    %minus = arith.constant dense<-3> : tensor<8xi16>
    %aa = arith.muli %a, %two : tensor<8xi16>
    %bb = arith.muli %b, %minus : tensor<8xi16>
    %sum = arith.addi %aa, %bb : tensor<8xi16>
    %y = arith.addi %sum, %c : tensor<8xi16>
    return %y : tensor<8xi16>
  }
}
