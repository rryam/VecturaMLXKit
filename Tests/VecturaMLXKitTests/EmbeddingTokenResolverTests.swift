import Foundation
import Testing
@testable import VecturaMLXKit

@Suite("Embedding Token Resolver")
struct EmbeddingTokenResolverTests {
  @Test("Padding token fallback order is eos -> unknown -> bos -> zero")
  func paddingTokenFallbackOrder() {
    #expect(
      EmbeddingTokenResolver.paddingTokenID(
        eosTokenID: 9,
        unknownTokenID: 3,
        bosTokenID: 1
      ) == 9
    )
    #expect(
      EmbeddingTokenResolver.paddingTokenID(
        eosTokenID: nil,
        unknownTokenID: 3,
        bosTokenID: 1
      ) == 3
    )
    #expect(
      EmbeddingTokenResolver.paddingTokenID(
        eosTokenID: nil,
        unknownTokenID: nil,
        bosTokenID: 1
      ) == 1
    )
    #expect(
      EmbeddingTokenResolver.paddingTokenID(
        eosTokenID: nil,
        unknownTokenID: nil,
        bosTokenID: nil
      ) == 0
    )
  }

  @Test("Attention mask rows follow sequence lengths")
  func attentionMaskRowsFollowLengths() {
    let rows = EmbeddingTokenResolver.attentionMaskRows(lengths: [3, 1, 0, 6], maxLength: 4)
    #expect(rows == [
      [1, 1, 1, 0],
      [1, 0, 0, 0],
      [0, 0, 0, 0],
      [1, 1, 1, 1],
    ])
  }
}
