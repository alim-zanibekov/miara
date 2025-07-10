// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Alim Zanibekov

const std = @import("std");
const w = @import("widow.zig");
const bit = @import("bit.zig");
const iterator = @import("iterator.zig");
const iface = @import("interface.zig");
const util = @import("util.zig");

pub const EliasFano = GenericEliasFano(u64, false, false, null);
pub const EliasFanoPS = GenericEliasFano(u64, false, true, null);
pub const EliasFanoGEQ = GenericEliasFano(u64, true, false, null);

/// Elias–Fano representation for monotonically increasing sequences
/// `T` - an unsigned integer type
/// `EnableGEQ` - enable get next greater or equal queries
/// `PrefixSumMode` - encode as prefix sum for arbitrary arrays (though it may overflow)
/// `CustomWidow` - custom Widow type, maybe configured with different params
pub fn GenericEliasFano(
    T: type,
    EnableGEQ: bool,
    PrefixSumMode: bool,
    CustomWidow: ?type,
) type {
    if (@typeInfo(T) != .int or @typeInfo(T).int.signedness != .unsigned)
        @compileError("EliasFano requires an unsigned integer type, found " ++ @typeName(T));

    const Widow = CustomWidow orelse if (EnableGEQ) w.WidowS1S0NR else w.WidowS1NR;
    const BitArray = bit.GenericBitArray(u64);

    return struct {
        const Self = @This();
        pub const Error = error{ OutOfBounds, Undefined };

        higher_bits_lookup: Widow,
        higher_bits: BitArray,
        lower_bits: BitArray,
        low_n: u16,
        len: usize,

        /// Initializes Elias–Fano structure with values from `iter`
        /// `universe` is the maximum possible value in the sequence
        ///     (must be greater than or equal to the last element in the sequence)
        ///     if 0 is passed - will be calculated
        /// `TIter` must implement `Iterator(T)` see `/iterator.zig`
        pub fn init(allocator: std.mem.Allocator, universe: T, TIter: type, iter: TIter) !Self {
            iface.checkImplementsGuard(iterator.ReusableIterator(T), iface.UnPtr(TIter));
            const m = lb: {
                if (universe == 0) {
                    var acc: u64 = 0;
                    while (iter.next()) |val| acc += val;
                    iter.reset();
                    break :lb acc;
                } else {
                    break :lb universe;
                }
            };

            const n = if (comptime PrefixSumMode) iter.size() + 1 else iter.size();

            const log_m = util.log2IntCeilOrZero(T, m);
            const log_n = util.log2IntCeilOrZero(usize, n);
            const low_n = if (log_m > log_n) log_m - log_n else 0;

            const lb_n = n * low_n;
            const hb_n = (m >> @intCast(low_n)) + n;

            var hb = try BitArray.initCapacity(allocator, hb_n);
            errdefer hb.deinit(allocator);
            var lb = try BitArray.initCapacity(allocator, lb_n);
            errdefer lb.deinit(allocator);

            var bucket: T = 0;

            if (comptime PrefixSumMode) {
                hb.appendUIntAssumeCapacity(@as(u8, 1), 1);
                if (low_n > 0) lb.appendUIntAssumeCapacity(@as(T, 0), @intCast(low_n));
            }

            var max: T = 0;
            var acc: T = 0;
            while (iter.next()) |val| {
                const value = if (comptime PrefixSumMode) lb: {
                    acc, const overflow = @addWithOverflow(acc, val);
                    if (overflow != 0) return error.Overflow;
                    break :lb acc;
                } else lb: {
                    max = @max(max, val);
                    if (max != val) return error.NonMonotomicallyIncreasing;
                    break :lb val;
                };

                const higher_bits = value >> @intCast(low_n);
                while (higher_bits > bucket) {
                    hb.appendUIntAssumeCapacity(@as(u8, 0), 1);
                    bucket += 1;
                }
                hb.appendUIntAssumeCapacity(@as(u8, 1), 1);
                if (low_n > 0) lb.appendUIntAssumeCapacity(value, @intCast(low_n));
            }

            const widow = try Widow.init(allocator, hb.len, hb.data);
            return Self{
                .lower_bits = lb,
                .higher_bits = hb,
                .higher_bits_lookup = widow,
                .low_n = low_n,
                .len = iter.size(),
            };
        }

        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            self.higher_bits_lookup.deinit(allocator);
            self.higher_bits.deinit(allocator);
            self.lower_bits.deinit(allocator);
            self.* = undefined;
        }

        /// Returns the i-th element in array
        pub fn get(self: *const Self, i: usize) !T {
            if (comptime PrefixSumMode) {
                return self.getDiff(i);
            } else {
                return self.getValue(i);
            }
        }

        /// Returns the i-th element of the array regardless of PrefixSumMode
        pub fn getValue(self: *const Self, i: usize) !T {
            const h_bits = try self.higher_bits_lookup.select1(i + 1) - i;
            const l_bits = lb: {
                if (self.low_n == 0) break :lb 0;
                if (comptime @bitSizeOf(T) <= @bitSizeOf(BitArray.Type)) {
                    break :lb self.lower_bits.getVarFast(self.low_n * i, @intCast(self.low_n));
                } else {
                    break :lb try self.lower_bits.getVar(T, @intCast(self.low_n), self.low_n * i);
                }
            };
            return (h_bits << @intCast(self.low_n)) | l_bits;
        }

        /// Returns the difference between the i + 1-th and i-th elements of the array
        pub fn getDiff(self: *const Self, i: usize) !T {
            const l_bits_1, const l_bits_2 = lb: {
                if (self.low_n == 0) break :lb .{ 0, 0 };
                if (comptime @bitSizeOf(T) <= @bitSizeOf(BitArray.Type)) {
                    const l_bits_1 = self.lower_bits.getVarFast(self.low_n * i, @intCast(self.low_n));
                    const l_bits_2 = self.lower_bits.getVarFast(self.low_n * (i + 1), @intCast(self.low_n));
                    break :lb .{ l_bits_1, l_bits_2 };
                } else {
                    const l_bits_1 = try self.lower_bits.getVar(T, @intCast(self.low_n), self.low_n * i);
                    const l_bits_2 = try self.lower_bits.getVar(T, @intCast(self.low_n), self.low_n * (i + 1));
                    break :lb .{ l_bits_1, l_bits_2 };
                }
            };

            const h_bits_1 = try self.higher_bits_lookup.select1(i + 1);
            const h_bits_2 = self.higher_bits.idxNext(h_bits_1 + 1, 1) orelse return error.NotFound;

            const v1 = ((h_bits_1 - i) << @intCast(self.low_n) | l_bits_1);
            const v2 = ((h_bits_2 - i - 1) << @intCast(self.low_n) | l_bits_2);

            return v2 - v1;
        }

        /// Finds the next value greater than or equal to `num`
        pub fn getNextGEQ(self: *const Self, num: T) !T {
            if (comptime !EnableGEQ or PrefixSumMode) {
                @panic("getNextGEQ is not available, enable it by setting the EnableGEQ flag or by disabling PrefixSumMode");
            }
            const i = num >> @intCast(self.low_n);
            var pos = if (i > 0) try self.higher_bits_lookup.select0(i) -| i else 0;
            var result = try self.getValue(pos);
            while (result < num and pos < self.len) : (pos += 1) {
                result = try self.getValue(pos);
            }
            return if (result < num) error.NotFound else result;
        }
    };
}

const testing = std.testing;

test "Elias Fano: get and getNextGEQ" {
    const n = 5000;
    const seed = 0x4C27A681B1;
    var r = std.Random.DefaultPrng.init(seed);
    var random = r.random();

    var data = try testing.allocator.alloc(u64, n);
    defer testing.allocator.free(data);

    data[0] = random.uintLessThan(u64, ~@as(u32, 0) / n);
    for (1..n) |i| {
        data[i] = data[i - 1] + random.uintLessThan(u64, ~@as(u32, 0) / n);
    }

    var iter = iterator.SliceIterator(u64).init(data);
    var ef = try EliasFanoGEQ.init(testing.allocator, data[data.len - 1], @TypeOf(&iter), &iter);
    defer ef.deinit(testing.allocator);

    for (data, 0..) |it, i| {
        try testing.expectEqual(it, ef.get(i));
    }

    for (0..n - 1) |i| {
        const goq = try ef.getNextGEQ(data[i]);
        try testing.expect(data[i + 1] == goq or data[i] == goq);
    }
}

test "Elias Fano: prefix sum" {
    const n = 5000;
    const seed = 0x4C27A681B1;
    var r = std.Random.DefaultPrng.init(seed);
    var random = r.random();

    const data = try testing.allocator.alloc(u64, n);
    defer testing.allocator.free(data);

    for (data) |*it| {
        it.* = random.uintLessThan(u64, ~@as(u32, 0));
    }

    var iter = iterator.SliceIterator(u64).init(data);
    var ef = try EliasFanoPS.init(testing.allocator, 0, @TypeOf(&iter), &iter);
    defer ef.deinit(testing.allocator);

    for (data, 0..) |it, i| {
        try testing.expectEqual(it, ef.get(i));
    }
}

test "Elias Fano: extreme case 1" {
    const n = 5000;
    const data = try testing.allocator.alloc(u64, n);
    defer testing.allocator.free(data);
    @memset(data, 0);
    data[data.len - 1] = ~@as(u64, 0);

    var iter = iterator.SliceIterator(u64).init(data);
    var ef = try EliasFano.init(testing.allocator, 0, @TypeOf(&iter), &iter);
    defer ef.deinit(testing.allocator);

    for (data, 0..) |it, i| {
        try testing.expectEqual(it, ef.get(i));
    }
}

test "Elias Fano: extreme case 2" {
    const n = 1;
    const data = try testing.allocator.alloc(u64, n);
    defer testing.allocator.free(data);
    @memset(data, 0);

    var iter = iterator.SliceIterator(u64).init(data);
    var ef = try EliasFano.init(testing.allocator, 0, @TypeOf(&iter), &iter);
    defer ef.deinit(testing.allocator);

    for (data, 0..) |it, i| {
        try testing.expectEqual(it, ef.get(i));
    }
}
