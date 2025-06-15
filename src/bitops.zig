const std = @import("std");

pub const NthSetBitError = error{ InvalidN, NotFound };

/// Returns the index (0-based, from msb) of the n-th set bit in `src`
/// Requires `src` to be an unsigned integer
pub fn nthSetBitPos(src: anytype, n: std.math.Log2IntCeil(@TypeOf(src))) NthSetBitError!std.math.Log2Int(@TypeOf(src)) {
    const T = @TypeOf(src);
    if (@typeInfo(T) != .int or @typeInfo(T).int.signedness != .unsigned)
        @compileError("nthSetBitPos requires an unsigned integer, found " ++ @typeName(T));

    if (n == 0 or n > @typeInfo(T).int.bits) {
        return NthSetBitError.InvalidN;
    }

    var count: usize = 0;
    inline for (0..@typeInfo(T).int.bits) |i| {
        if (src & (@as(T, 1) << @intCast(@typeInfo(T).int.bits - 1 - i)) > 0) {
            count += 1;
            if (count == n) {
                return @intCast(i);
            }
        }
    }

    return error.NotFound;
}

const testing = std.testing;

test "nthSetBitPos" {
    try testing.expectEqual(0, nthSetBitPos(~@as(u64, 0), 1));
    try testing.expectEqual(63, nthSetBitPos(@as(u64, 1), 1));
    try testing.expectEqual(8, nthSetBitPos(@as(u9, 1), 1));
    try testing.expectEqual(62, nthSetBitPos(@as(u64, 7), 2));
    try testing.expectEqual(0, nthSetBitPos(@as(u1, 1), 1));
    try testing.expectEqual(1, nthSetBitPos(@as(u2, 1), 1));
    try testing.expectEqual(2, nthSetBitPos(@as(u3, 7), 3));
    try testing.expectEqual(63, nthSetBitPos(~@as(u64, 0), 64));
    try testing.expectEqual(14, nthSetBitPos(~@as(u64, 0), 15));

    try testing.expectError(NthSetBitError.InvalidN, nthSetBitPos(@as(u64, 1), 0));
    try testing.expectError(NthSetBitError.NotFound, nthSetBitPos(@as(u64, 1), 2));
    try testing.expectError(NthSetBitError.NotFound, nthSetBitPos(@as(u2, 1), 2));
    try testing.expectError(NthSetBitError.InvalidN, nthSetBitPos(~@as(u64, 0), 65));
}
