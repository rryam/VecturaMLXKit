import Foundation

enum EmbeddingTokenResolver {
  static func paddingTokenID(
    eosTokenID: Int?,
    unknownTokenID: Int?,
    bosTokenID: Int?
  ) -> Int {
    eosTokenID ?? unknownTokenID ?? bosTokenID ?? 0
  }

  static func attentionMaskRows(lengths: [Int], maxLength: Int) -> [[Bool]] {
    lengths.map { length in
      let clampedLength = min(max(0, length), maxLength)
      let paddingCount = max(0, maxLength - clampedLength)
      return Array(repeating: true, count: clampedLength)
        + Array(repeating: false, count: paddingCount)
    }
  }
}
