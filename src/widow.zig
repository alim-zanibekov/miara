const std = @import("std");
const builtin = @import("builtin");
const util = @import("util.zig");
const bit = @import("bit.zig");

inline fn fastdivPrecompute(divisor: u64) u128 {
    return (~@as(u128, 0) / @as(u128, divisor)) + 1;
}

inline fn fastdiv(dividend: u64, precomputed: u128) u64 {
    const bottom_half = ((precomputed & ~@as(u64, 0)) * @as(u128, dividend)) >> 64;
    const top_half = (precomputed >> 64) * @as(u128, dividend);
    return @intCast((bottom_half + top_half) >> 64);
}

inline fn fastmod(dividend: u64, precomputed: u128, divisor: u64) u64 {
    const p = precomputed *% @as(u128, dividend);
    return fastdiv(divisor, p);
}

pub fn PowerOf2Int(comptime T: type) type {
    const info = @typeInfo(T).int;
    const bits = std.math.ceilPowerOfTwo(u64, info.bits) catch unreachable;
    return std.meta.Int(info.signedness, bits);
}

/// Full-featured. Select and rank support
pub const Widow = GenericWidow(u64, true, true, &[_]u64{ 1 << 16, 1 << 9 }, &[_]usize{ 1 << 16, 1 << 9 }, true, false);
/// Only select 1 and rank. Select 0 is disabld
pub const WidowS1 = GenericWidow(u64, true, false, &[_]u64{ 1 << 16, 1 << 9 }, &[_]usize{ 1 << 16, 1 << 9 }, true, false);
/// Select 1 without rank support, level constants for Elias Fano (for dense, will overlow on sparse arrays)
pub const WidowS1NR = GenericWidow(u64, true, false, &[_]u64{}, &[_]usize{ 8192, 64 }, false, true);
/// Select 1/0 without rank support, level constants for Elias Fano with GEQ support (for dense, will overlow on sparse arrays)
pub const WidowS1S0NR = GenericWidow(u64, true, true, &[_]u64{}, &[_]usize{ 8192, 64 }, false, true);
/// Rank-only
pub const WidowR = GenericWidow(u64, false, false, &[_]u64{ 1 << 16, 1 << 9 }, &[_]usize{}, true, false);

/// Data structure for efficient bit-based rank and select queries.
/// Based on SPIDER architecture, but features a more sophisticated select 1/0 lookup
/// structure and enhanced configurability. The versions used in EliasFano are more
/// similar to the DArray data structure in terms of runtime performance, taking about
/// 2 iterations to locate the requested bit
pub fn GenericWidow(
    /// The backing slice type
    comptime T: type,
    /// Enable select operations for 1-bits
    comptime Select1: bool,
    /// Enable select operations for 0-bits
    comptime Select0: bool,
    /// Hierarchy levels for the rank data structure (in bits)
    /// Must be powers of two
    comptime RankLevels: []const u64,
    /// Hierarchy levels for the select data structure (in bits)
    /// The offset types are calculated based on these numbers, for example
    /// if the levels are 2^16, 2^9 it will result in u64, u16, if the levels are 2^17, 2^9 it will
    /// result in u64, u32, and etc. For very sparse bit arrays, levels will be
    /// automatically recalculated with appropriate scaling
    comptime SelectLevels: []const u64,
    /// Use the rank data structure to accelerate select queries
    /// This approach requires less precision for select data types
    comptime SelectUsingRank: bool,
    /// Disable rank operations entirely
    comptime NoRank: bool,
) type {
    const BitArray = []const T;
    const BitSize = u64;
    const worst_len = ~@as(BitSize, 0);
    const SparseThreshold: f64 = 0.4;

    if ((Select1 or Select0) and NoRank and SelectUsingRank) {
        @compileError("Widow: incompatible parameters 'NoRank' and 'SelectUsingRank' are both true");
    }

    for (RankLevels) |it| if (!std.math.isPowerOfTwo(it)) @compileError("RankLevels values must be powers of two");

    const RankSelectDivisor = @bitSizeOf(T) / 2;

    return struct {
        const Self = @This();

        pub const Type = T;

        pub const RankLevelTypes = lb: {
            var types = [_]type{BitSize} ** RankLevels.len;
            types[0] = PowerOf2Int(std.math.IntFittingRange(0, worst_len - 1));
            for (1..RankLevels.len) |i| {
                types[i] = PowerOf2Int(std.math.IntFittingRange(0, RankLevels[i - 1] - 1));
            }
            break :lb types;
        };

        pub const SelectLevelTypes = lb: {
            var types = [_]type{BitSize} ** SelectLevels.len;
            types[0] = PowerOf2Int(std.math.IntFittingRange(0, worst_len - 1));
            for (1..SelectLevels.len) |i| {
                types[i] = PowerOf2Int(std.math.IntFittingRange(0, SelectLevels[i - 1] - 1));
            }
            break :lb types;
        };

        pub const SelectTable = std.meta.Tuple(lb: {
            var types = [_]type{BitSize} ** SelectLevels.len;
            for (0..SelectLevels.len) |i| types[i] = []SelectLevelTypes[i];
            break :lb &types;
        });

        pub const RankTable = std.meta.Tuple(lb: {
            var types = [_]type{BitSize} ** RankLevels.len;
            for (0..RankLevels.len) |i| types[i] = []RankLevelTypes[i];
            break :lb &types;
        });

        pub const SelectSupport = struct {
            strides: [SelectLevels.len]usize,
            strides_div: [SelectLevels.len]u128,
            table: SelectTable,
            size: BitSize,

            fn deinit(self: *@This(), allocator: std.mem.Allocator) void {
                inline for (0..SelectLevelTypes.len) |i| {
                    if (self.table[i].len > 0) allocator.free(self.table[i]);
                }
                self.* = undefined;
            }
        };

        pub const RankSupport = struct {
            table: RankTable,
            n_set: BitSize,
            size: BitSize,

            fn deinit(self: *@This(), allocator: std.mem.Allocator) void {
                inline for (0..RankLevelTypes.len) |i| {
                    if (self.table[i].len > 0) allocator.free(self.table[i]);
                }
                self.* = undefined;
            }
        };

        bit_array: BitArray,
        rank_table: if (NoRank) void else RankSupport,
        select1_table: if (Select1) SelectSupport else void,
        select0_table: if (Select0) SelectSupport else void,

        /// Initialize the data structure with the given bit array
        pub fn init(
            allocator: std.mem.Allocator,
            bit_len: BitSize,
            bit_array: BitArray,
        ) !Self {
            // std.debug.print("{s}\n", .{std.fmt.comptimePrint("{any}", .{SelectLevelTypes})});

            if (bit_len == 0) return error.BitArrayIsEmpty;
            if (bit_array.len == 0) return error.BitArrayIsEmpty;
            std.debug.assert(bit_len <= bit_array.len * @bitSizeOf(T));

            var rank_table = if (!NoRank) try buildRank(allocator, bit_len, bit_array) else {};
            errdefer if (!NoRank) rank_table.deinit(allocator) else {};

            const n_set = if (NoRank) lb: {
                var sum: BitSize = 0;
                const len = (bit_len - 1) / @bitSizeOf(T) + 1;
                for (bit_array[0 .. len - 1]) |it| sum += @popCount(it);
                const bit_size: BitSize = len * @bitSizeOf(T);
                const count = @popCount(bit_array[len - 1] >> @intCast(bit_size - bit_len));
                break :lb sum + count;
            } else rank_table.n_set;

            var select1_table = if (Select1) try buildSelect(allocator, n_set, bit_len, bit_array, false) else {};
            errdefer if (Select1) select1_table.deinit(allocator) else {};

            var select0_table = if (Select0) try buildSelect(allocator, n_set, bit_len, bit_array, true) else {};
            errdefer if (Select0) select0_table.deinit(allocator) else {};

            return Self{
                .bit_array = bit_array,
                .rank_table = rank_table,
                .select1_table = select1_table,
                .select0_table = select0_table,
            };
        }

        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            if (!NoRank) self.rank_table.deinit(allocator);
            if (Select0) self.select0_table.deinit(allocator);
            if (Select1) self.select1_table.deinit(allocator);
            self.* = undefined;
        }

        /// Build hierarchical ranking structure for efficient bit counting queries
        pub fn buildRank(
            allocator: std.mem.Allocator,
            bit_len: BitSize,
            bit_array: BitArray,
        ) !RankSupport {
            std.debug.assert(bit_len > 0);
            std.debug.assert(bit_len <= bit_array.len * @bitSizeOf(T));

            var rs = RankSupport{
                .table = .{&.{}} ** RankLevelTypes.len,
                .n_set = 0,
                .size = bit_len,
            };
            errdefer rs.deinit(allocator);

            inline for (RankLevels, 0..) |size, i| {
                rs.table[i] = try allocator.alloc(RankLevelTypes[i], ((bit_len - 1) / size) + 1 + 2);
                rs.table[i][0] = 0;
            }

            var cumulative_count: BitSize = 0;
            const len = (bit_len - 1) / @bitSizeOf(T) + 1;
            for (bit_array[0 .. len - 1], 0..) |value, pos| {
                const count = @popCount(value);
                const bit_pos: BitSize = (pos + 1) * @bitSizeOf(@TypeOf(value));
                cumulative_count += count;

                inline for (RankLevels, 0..) |stride, i| {
                    if (bit_pos % stride == 0) {
                        if (comptime i > 0) {
                            const j_prev = bit_pos / RankLevels[i - 1];
                            rs.table[i][bit_pos / stride] = @intCast(cumulative_count - rs.table[i - 1][j_prev]);
                        } else {
                            rs.table[i][bit_pos / stride] = cumulative_count;
                        }
                    }
                }
            }

            {
                const bit_size: BitSize = len * @bitSizeOf(T);
                const count = @popCount(bit_array[len - 1] >> @intCast(bit_size - bit_len));
                cumulative_count += count;

                const bit_pos = bit_len;
                inline for (RankLevels, 0..) |stride, i| {
                    if (comptime i > 0) {
                        const j_prev = bit_pos / RankLevels[i - 1];
                        rs.table[i][(bit_pos - 1) / stride + 1] = @intCast(cumulative_count - rs.table[i - 1][j_prev]);
                    } else {
                        rs.table[i][(bit_pos - 1) / stride + 1] = cumulative_count;
                    }
                }
            }
            rs.n_set = cumulative_count;
            return rs;
        }

        /// Build hierarchical select support structure for efficient bit position queries
        pub fn buildSelect(
            allocator: std.mem.Allocator,
            /// The number of set bits (1s) in the array, regardless of the flip
            n_set: BitSize,
            bit_len: BitSize,
            bit_array: BitArray,
            /// True for select 0 queries, false for select 1 queries
            comptime flip: bool,
        ) !SelectSupport {
            std.debug.assert(bit_len > 0);
            const n_select = SelectLevels.len;
            const num_set = if (flip) bit_len - n_set else n_set;

            var ss = SelectSupport{
                .table = .{&.{}} ** n_select,
                .strides = .{0} ** n_select,
                .strides_div = .{0} ** n_select,
                .size = num_set,
            };
            errdefer ss.deinit(allocator);

            var counts: [n_select]BitSize = .{0} ** n_select;

            const l0_size = (SelectLevels[0] * num_set) / bit_len;

            if (SparseThreshold * @as(f64, @floatFromInt(bit_len)) > @as(f64, @floatFromInt(num_set))) {
                if (multiplicativeChain(allocator, BitSize, l0_size, SelectLevels) catch null) |strides_x| {
                    @memcpy(&ss.strides, strides_x);
                    allocator.free(strides_x);
                } else {
                    ss.strides = .{@bitSizeOf(T) * 2} ** n_select;
                }
            } else {
                @memcpy(&ss.strides, SelectLevels);
            }

            inline for (&ss.strides, 0..) |block, i| ss.strides_div[i] = fastdivPrecompute(block);

            inline for (&ss.strides, 0..) |stride, i| {
                ss.table[i] = try allocator.alloc(SelectLevelTypes[i], if (num_set == 0) 2 else ((num_set - 1) / stride + 1) + 2);
                ss.table[i][0] = 0;
                ss.table[i][ss.table[i].len - 1] = 0;
            }

            if (num_set == 0) {
                return ss;
            }

            var idx: usize = 0;
            while (idx < bit_array.len) : (idx += 1) {
                const value = if (flip) ~bit_array[idx] else bit_array[idx];
                if (@popCount(value) != 0) {
                    const first_1_pos = idx * @bitSizeOf(T) + try bit.nthSetBitPos(value, 1);
                    if (SelectUsingRank) {
                        ss.table[0][0] = first_1_pos / RankSelectDivisor;
                    } else {
                        ss.table[0][0] = first_1_pos;
                    }
                    break;
                }
            }
            std.debug.assert(idx < bit_array.len);

            var pos: BitSize = 0;
            var cumulative_count: BitSize = 0;
            for (bit_array[idx..]) |value_raw| {
                const value = if (flip) ~value_raw else value_raw;
                const count = @popCount(value);
                inline for (&ss.strides, 0..) |stride, i| {
                    if (counts[i] + count < stride) {
                        counts[i] += count;
                    } else {
                        std.debug.assert(counts[i] < stride);

                        var curr_value = value;
                        var curr_acc = cumulative_count;
                        var curr_pos = pos;
                        var curr_count = count;

                        while (counts[i] + curr_count >= stride) {
                            const remaining = stride - counts[i];
                            std.debug.assert(remaining != 0);
                            const value_pos: u16 = try bit.nthSetBitPos(curr_value, @intCast(remaining));
                            const table_pos = fastdiv((curr_acc + remaining), ss.strides_div[i]);
                            if (comptime i == 0) {
                                if (SelectUsingRank) {
                                    ss.table[i][table_pos] = (curr_pos + value_pos) / RankSelectDivisor;
                                } else {
                                    ss.table[i][table_pos] = curr_pos + value_pos;
                                }
                            } else {
                                const prev_level_pos = fastdiv((curr_acc + remaining), ss.strides_div[i - 1]);
                                const treshold = comptime std.math.maxInt(SelectLevelTypes[i]);
                                if (SelectUsingRank) {
                                    const rel_pos = ((curr_pos + value_pos) / RankSelectDivisor) - ss.table[i - i][prev_level_pos];
                                    ss.table[i][table_pos] = if (rel_pos > treshold) treshold else @intCast(rel_pos);
                                } else {
                                    const rel_pos = (curr_pos + value_pos) - ss.table[i - i][prev_level_pos];
                                    if (rel_pos > treshold) return error.Overflow;
                                    ss.table[i][table_pos] = @intCast(rel_pos);
                                }
                            }

                            curr_value = bit.shl(curr_value, value_pos + 1);
                            curr_count -= @intCast(remaining);
                            curr_acc += remaining;
                            curr_pos += value_pos + 1;

                            counts[i] = 0;
                        }
                        counts[i] = curr_count;
                    }
                }
                cumulative_count += count;
                pos += @bitSizeOf(T);
            }

            return ss;
        }

        /// Returns the number of 1 bits up to and including index `i` (0 < i < bit_len)
        pub fn rank1(self: *Self, i: u64) !u64 {
            if (NoRank) @compileError("rank1 is not available, enable it by setting the NoRank flag to false");
            if (i == 0 or i > self.rank_table.size) return error.OutOfBounds;

            return rank(self.rank_table, self.bit_array, i, 1);
        }

        /// Returns the number of 0 bits up to and including index `i` (0 < i < bit_len)
        pub fn rank0(self: *Self, i: u64) !u64 {
            if (NoRank) @compileError("rank0 is not available, enable it by setting the NoRank flag to false");
            if (i == 0 or i > self.rank_table.size) return error.OutOfBounds;

            return rank(self.rank_table, self.bit_array, i, 0);
        }

        /// Returns the number of `needle` bits up to and including index `i` (0 < i < bit_len)
        pub fn rank(
            rs: RankSupport,
            bit_array: BitArray,
            i: BitSize,
            comptime needle: u1,
        ) BitSize {
            var count: BitSize = 0;
            inline for (0.., RankLevels) |n, stride| {
                count += rs.table[n][i / stride];
            }
            const last_stride = comptime RankLevels[RankLevels.len - 1];
            const begin = ((i / last_stride) * last_stride) / @bitSizeOf(T);
            const end = i / @bitSizeOf(T);

            for (bit_array[begin..end]) |it| {
                count += @popCount(it);
            }

            const remaining: std.math.Log2Int(T) = @intCast(i - (end * @bitSizeOf(T)));
            if (remaining > 0) {
                // @intFromBool(remaining > 0) mul?
                count += @popCount(bit_array[end] >> @intCast(@as(std.math.Log2IntCeil(T), @bitSizeOf(T)) - remaining));
            }

            if (comptime needle == 1) {
                return count;
            } else {
                return i - count;
            }
        }

        /// Returns the position of the i-th 1-bit. `i` is a 1-based index; result is a 0-based index
        pub fn select1(self: *const Self, i: u64) !u64 {
            if (!Select1) unreachable;
            if (i == 0 or i > self.select1_table.size) return error.OutOfBounds;
            return select(self.rank_table, self.select1_table, self.bit_array, i, 1);
        }

        /// Returns the position of the i-th 0-bit. `i` is a 1-based index; result is a 0-based index
        pub fn select0(self: *const Self, i: u64) !u64 {
            if (!Select0) unreachable;
            if (i == 0 or i > self.select0_table.size) return error.OutOfBounds;
            return select(self.rank_table, self.select0_table, self.bit_array, i, 0);
        }

        /// Returns the position of the i-th `needle` bit. `i` is a 1-based index; result is a 0-based index
        pub noinline fn select(
            rs: if (NoRank) void else RankSupport,
            st: SelectSupport,
            bit_array: BitArray,
            i: BitSize,
            comptime needle: u1,
        ) !u64 {
            comptime if (!Select0 and needle == 0) @compileError("select0 is not available, enable it by setting the Select0 flag");
            comptime if (!Select1 and needle == 1) @compileError("select1 is not available, enable it by setting the Select1 flag");

            var bit_pos: BitSize = 0;
            inline for (&st.table, &st.strides_div) |row, precomputed| {
                if (comptime SelectUsingRank) {
                    bit_pos += row[fastdiv(i, precomputed)] * RankSelectDivisor;
                } else {
                    bit_pos += row[fastdiv(i, precomputed)];
                }
            }

            var remaining: BitSize = undefined;
            const sll = SelectLevels.len;
            if (comptime SelectUsingRank) {
                remaining = i - rank(rs, bit_array, bit_pos, needle);
            } else {
                remaining = fastmod(i, st.strides_div[sll - 1], st.strides[sll - 1]);
                if (i >= st.strides[sll - 1]) remaining += 1;
            }

            if (remaining == 0) {
                return bit_pos;
            }

            var idx = bit_pos / @bitSizeOf(T);
            const shift = bit_pos % @bitSizeOf(T);
            var value: T = (if (comptime needle == 0) ~bit_array[idx] else bit_array[idx]) & (~@as(T, 0) >> @intCast(shift));

            while (true) {
                const pcnt = @popCount(value);
                if (remaining <= pcnt) {
                    const value_pos = try bit.nthSetBitPos(value, @intCast(remaining));
                    return (idx * @bitSizeOf(T)) + value_pos;
                }
                remaining -= pcnt;
                idx += 1;
                value = if (comptime needle == 0) ~bit_array[idx] else bit_array[idx];
            }

            unreachable;
        }
    };
}

/// Computes an optimal multiplicative `chain` starting from `init` using the given `hints`.
/// The function attempts to construct a sequence a1, a2, a3 .. a[hints.len -1] such that
/// - a1 <= init
/// - Each a[i] divides a[i - 1]
/// - The shape of the `chain` roughly follows the scale suggested by `hints`
/// Always returns a new allocated slice, even if `hints` is an empty array
pub fn multiplicativeChain(allocator: std.mem.Allocator, T: type, init: T, hints: []const T) ![]const T {
    if (hints.len <= 1) {
        const out = try allocator.alloc(u64, 1);
        out[0] = init;
        return out;
    }

    var divisors_chain = try allocator.alloc(T, hints.len - 1);
    defer allocator.free(divisors_chain);

    var curr = hints[0];
    for (divisors_chain, 1..) |*d, i| {
        d.* = @max(2, curr / hints[i]);
        curr /= d.*;
    }

    while (true) {
        var total: u64 = 1;
        for (divisors_chain) |d| total *= d;

        const ref = (init / total) * total;
        if (ref == 0) {
            var max_index: usize = 0;
            for (divisors_chain, 0..) |d, i| {
                if (d > divisors_chain[max_index]) max_index = i;
            }

            if (divisors_chain[max_index] > 2) {
                divisors_chain[max_index] -= 1;
                continue;
            } else {
                return error.NotFound;
            }
        }

        const out = try allocator.alloc(u64, hints.len);
        out[0] = ref;
        for (divisors_chain, 0..) |d, i| {
            out[i + 1] = out[i] / d;
        }
        return out;
    }
}

const testing = std.testing;

test multiplicativeChain {
    const allocator = testing.allocator;
    {
        const hints = [_]u64{ 10000, 4000, 1400, 100, 2 };
        const expected = [_]u64{ 9984, 4992, 1664, 104, 2 };

        const result = try multiplicativeChain(allocator, u64, hints[0], &hints);
        defer allocator.free(result);

        try testing.expectEqualSlices(u64, &expected, result);
    }
    {
        const hints = [_]u64{ 9973, 3000, 1000, 250, 5 };
        const result = try multiplicativeChain(allocator, u64, hints[0], &hints);
        defer allocator.free(result);

        try testing.expect(result.len == 5);
        for (result[1..], 1..) |v, i| {
            try testing.expect(result[i - 1] % v == 0);
            try testing.expect(v <= result[i - 1] / 2);
        }
    }
    {
        const hints = [_]u64{12345};
        const result = try multiplicativeChain(allocator, u64, hints[0], &hints);
        defer allocator.free(result);
        try testing.expectEqualSlices(u64, &hints, result);
    }
    {
        const hints = [_]u64{ 3, 2, 1 };
        const result = multiplicativeChain(allocator, u64, hints[0], &hints);
        try testing.expectError(error.NotFound, result);
    }
    {
        const mu = 1 << 63;
        const hints = [_]u64{ mu, mu / 16, mu / 64, 1 };
        const result = try multiplicativeChain(allocator, u64, hints[0], &hints);
        defer allocator.free(result);
        try testing.expectEqualSlices(u64, &hints, result);
    }
}

const WidowU8 = GenericWidow(u8, true, true, &[_]u64{ 65_536, 512 }, &[_]usize{ 65_536 + 1, 128 }, false, false);

test "widow: rank" {
    const seed = 0x71CD3A1E15;
    const buf_size = 1 << 20;

    var r = std.Random.DefaultPrng.init(seed);
    var random = r.random();

    var data = try testing.allocator.alloc(u8, buf_size);
    defer testing.allocator.free(data);

    data.len = buf_size;
    random.bytes(data);

    var widow = try WidowU8.init(testing.allocator, buf_size * 8, data);
    defer widow.deinit(testing.allocator);

    var count: u64 = 0;

    for (0..buf_size) |i| {
        const b = random.uintAtMost(u4, 8);
        const expectation = count + @popCount(@as(u16, data[i]) >> @intCast(8 - b));
        const rank = try widow.rank1((i * 8) + b);
        try testing.expectEqual(expectation, rank);
        count += @popCount(data[i]);
    }

    try testing.expectEqual(count, try widow.rank1(buf_size * 8));
    try testing.expectError(error.OutOfBounds, widow.rank1(0));
}

test "widow: select 1" {
    const seed = 0x4C27A681B1;
    const buf_size = 1 << 10;

    var r = std.Random.DefaultPrng.init(seed);
    var random = r.random();

    var data = try testing.allocator.alloc(u8, buf_size);
    defer testing.allocator.free(data);

    data.len = buf_size;
    random.bytes(data);

    var widow = try WidowU8.init(testing.allocator, buf_size * 8, data);
    defer widow.deinit(testing.allocator);

    var count: u64 = 0;

    for (0..buf_size) |i| {
        if (data[i] == 0) continue;
        const n = random.uintAtMost(u4, @popCount(data[i]) - 1) + 1;
        const pos = try bit.nthSetBitPos(data[i], n);
        const expectation = i * 8 + pos;

        const select = try widow.select1(count + n);
        try testing.expectEqual(expectation, select);

        count += @popCount(data[i]);
    }

    try testing.expectError(error.OutOfBounds, widow.select1(0));
    try testing.expectEqual(2, widow.select1(1));
}

test "widow: select 0" {
    const seed = 0x3D93F7A0B1;
    const buf_size = 1 << 10;

    var r = std.Random.DefaultPrng.init(seed);
    var random = r.random();

    var data = try testing.allocator.alloc(u8, buf_size);
    defer testing.allocator.free(data);

    data.len = buf_size;
    random.bytes(data);

    var widow = try WidowU8.init(testing.allocator, buf_size * 8, data);
    defer widow.deinit(testing.allocator);

    var count: u64 = 0;

    for (0..buf_size) |i| {
        const value = ~data[i];
        if (value == 0) continue;
        const n = random.uintAtMost(u4, @popCount(value) - 1) + 1;
        const pos = try bit.nthSetBitPos(value, n);
        const expectation = i * 8 + pos;
        const select = try widow.select0(count + n);
        try testing.expectEqual(expectation, select);

        count += @popCount(value);
    }

    try testing.expectError(error.OutOfBounds, widow.select1(0));
    try testing.expectEqual(2, widow.select1(1));
}

test "widow: extreme cases" {
    const seed = 0x4C471781B1;
    const buf_size = 1 << 20;

    const T = u64;

    var r = std.Random.DefaultPrng.init(seed);
    var random = r.random();

    var data = try testing.allocator.alloc(T, buf_size);
    defer testing.allocator.free(data);
    data.len = buf_size;
    data[0] = random.int(T);
    @memset(data[1..], 0);
    data[data.len - 1] = random.int(T);

    const len = buf_size * @bitSizeOf(T);

    var widow = try Widow.init(testing.allocator, len, data);
    defer widow.deinit(testing.allocator);

    {
        var count: u64 = 0;

        for (0..buf_size) |i| {
            const b = random.uintAtMost(T, @bitSizeOf(T) - 1) + 1;
            const expectation = count + @popCount(data[i] >> @intCast(@bitSizeOf(T) - b));
            const rank = widow.rank1((i * @bitSizeOf(T)) + b);
            try testing.expectEqual(expectation, rank);
            count += @popCount(data[i]);
        }

        try testing.expectEqual(count, try widow.rank1(buf_size * @bitSizeOf(T)));
        try testing.expectEqual(
            (data.len - 1) * @bitSizeOf(T) + try bit.nthSetBitPos(data[data.len - 1], @popCount(data[data.len - 1])),
            try widow.select1(count),
        );
    }

    {
        var count: u64 = 0;

        for (0..buf_size) |i| {
            const value = ~data[i];
            if (value == 0) continue;
            const n = random.uintAtMost(u7, @popCount(value) - 1) + 1;
            const pos = try bit.nthSetBitPos(value, n);
            const expectation = i * @bitSizeOf(T) + pos;
            const select = try widow.select0(count + n);
            try testing.expectEqual(expectation, select);

            count += @popCount(value);
        }
        try testing.expectEqual(count, try widow.rank0(buf_size * @bitSizeOf(T)));
        try testing.expectEqual(
            (data.len - 1) * @bitSizeOf(T) + try bit.nthSetBitPos(~data[data.len - 1], @popCount(~data[data.len - 1])),
            try widow.select0(count),
        );
    }

    {
        var count: u64 = 0;

        for (0..buf_size) |i| {
            if (data[i] == 0) continue;
            const n = random.uintAtMost(u7, @popCount(data[i]) - 1) + 1;
            const pos = try bit.nthSetBitPos(data[i], n);
            const expectation = i * 64 + pos;

            const select = try widow.select1(count + n);
            try testing.expectEqual(expectation, select);

            count += @popCount(data[i]);
        }
        try testing.expectEqual(count, try widow.rank1(buf_size * @bitSizeOf(T)));
    }

    try testing.expectError(error.OutOfBounds, widow.rank1(0));
}
